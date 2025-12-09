#!/usr/bin/env python3
import argparse
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# Dataset
# ============================================================
class PatchDataset(Dataset):
    def __init__(self, npz_path, normalize=True):

        data = np.load(npz_path, mmap_mode="r")
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

        # ---------------------------------------------------
        # NEW: Combine A & B → single channel (A+B)
        # X: (N, 2, H, W) → (N, 1, H, W)
        # ---------------------------------------------------
        if X.ndim == 4 and X.shape[1] == 2:
            X = X.sum(axis=1, keepdims=True)
            print("Combined A/B → single channel. New X shape:", X.shape)

        if normalize:
            max_val = np.max(X)
            if max_val > 0:
                X = X / max_val

        self.X = X
        self.y = y

        print("Loaded:", npz_path)
        print("  X shape:", self.X.shape)
        print("  y shape:", self.y.shape)
        print("  Class balance:", np.unique(self.y, return_counts=True))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


# ============================================================
# CNN Model
# ============================================================
class PatchCNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, dropout=0.3):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32→16

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16→8

            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),  # → (N, c3, 1,1)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    logits_list = []
    labels_list = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        logits_list.append(logits.cpu().numpy())
        labels_list.append(yb.cpu().numpy())

    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(labels, probs)
        fpr, tpr, thresholds = roc_curve(labels, probs)
    except:
        auc, fpr, tpr, thresholds = np.nan, None, None, None

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "confusion_matrix": confusion_matrix(labels, preds),
    }


def plot_roc_curve(metrics, outpath):
    if metrics["fpr"] is None:
        print("No ROC curve: only one class present.")
        return

    plt.figure()
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"AUC={metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.savefig(outpath)
    plt.close()
    print("Saved ROC plot:", outpath)


# ============================================================
# Training loop
# ============================================================
def train_model(train_loader, val_loader, dataset, device,
                epochs=10, lr=1e-3, base_channels=32, dropout=0.3):

    model = PatchCNN(in_channels=dataset.X.shape[1],
                     base_channels=base_channels,
                     dropout=dropout).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        total_loss /= len(train_loader.dataset)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss={total_loss:.4f} | "
            f"Train AUC={train_metrics['auc']:.4f} | "
            f"Val AUC={val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)

    return model, val_metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="AB_patches_32x32.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="cnn_results")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--grid-search", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)

    dataset = PatchDataset(args.data)
    y_all = dataset.y

    class_counts = np.bincount(y_all.astype(int))
    print("Class counts:", class_counts)

    # splits
    N = len(dataset)
    n_test = int(args.test_frac * N)
    n_val = int(args.val_frac * N)
    n_train = N - n_val - n_test

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    # Weighted sampler for imbalance
    y_train = y_all[train_ds.indices].astype(int)
    sample_weights = (1.0 / class_counts)[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def make_loaders(batch):
        return (
            DataLoader(train_ds, batch_size=batch, sampler=sampler),
            DataLoader(val_ds, batch_size=batch, shuffle=False),
            DataLoader(test_ds, batch_size=batch, shuffle=False)
        )

    # No grid search (default)
    train_loader, val_loader, test_loader = make_loaders(args.batch_size)

    model, _ = train_model(
        train_loader, val_loader, dataset, device,
        epochs=args.epochs,
        lr=args.lr,
        base_channels=args.base_channels,
        dropout=args.dropout,
    )

    test_metrics = evaluate(model, test_loader, device)

    print("\n=== FINAL TEST METRICS ===")
    for k, v in test_metrics.items():
        if k not in ["fpr", "tpr", "thresholds"]:
            print(f"{k}: {v}")

    plot_roc_curve(test_metrics, os.path.join(args.output_dir, "roc_curve.png"))
    np.savez(os.path.join(args.output_dir, "metrics.npz"), **test_metrics)
    print("Saved:", os.path.join(args.output_dir, "metrics.npz"))


if __name__ == "__main__":
    main()
