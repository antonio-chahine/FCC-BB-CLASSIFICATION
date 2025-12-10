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
    roc_auc_score, roc_curve, accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)


# ============================================================
# Dataset
# ============================================================
class PatchDataset(Dataset):
    def __init__(self, npz_path, normalize=True):
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

        # Keep A/B separate
        print("A/B channels:", X.shape)

        if normalize:
            max_val = np.max(X)
            if max_val > 0:
                X = X / max_val

        self.X = X
        self.y = y

        print("Loaded:", npz_path)
        print("X shape:", X.shape)
        print("Class balance:", np.unique(y, return_counts=True))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


# ============================================================
# CNN Model
# ============================================================
class PatchCNN(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, dropout=0.3):
        super().__init__()

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
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

    logits_all, labels_all = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        logits_all.append(logits.cpu().numpy())
        labels_all.append(yb.cpu().numpy())

    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)

    try:
        auc = roc_auc_score(labels, probs)
        fpr, tpr, _ = roc_curve(labels, probs)
    except:
        auc, fpr, tpr = np.nan, None, None

    return {
        "auc": auc, "acc": acc, "precision": precision, "recall": recall,
        "f1": f1, "confusion_matrix": cm, "fpr": fpr, "tpr": tpr,
        "loss": None,
    }


# ============================================================
# Training Loop
# ============================================================
def train(model, train_loader, val_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_aucs = [], []
    best_auc, best_state = -1, None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        total_loss /= len(train_loader.dataset)
        train_losses.append(total_loss)

        val_metrics = evaluate(model, val_loader, device)
        val_aucs.append(val_metrics["auc"])

        print(f"Epoch {epoch} Loss={total_loss:.4f} Val AUC={val_metrics['auc']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, train_losses, val_aucs


# ============================================================
# Plotting
# ============================================================
def plot_loss(train_losses, outpath):
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(outpath)
    plt.close()


def plot_roc(metrics, outpath):
    if metrics["fpr"] is None:
        print("No ROC available.")
        return

    plt.figure()
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"AUC={metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True)
    plt.legend()
    plt.savefig(outpath)
    plt.close()


# ============================================================
# Hyperparameter Search
# ============================================================
def grid_search(hparams, dataset, device, train_ds, val_ds, batch_size):
    best_auc = -1
    best_model = None
    best_config = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for base_channels, dropout, lr in itertools.product(
        hparams["base_channels"], hparams["dropout"], hparams["lr"]
    ):
        print(f"\nTesting config: base={base_channels} drop={dropout} lr={lr}")

        model = PatchCNN(
            in_channels=dataset.X.shape[1],
            base_channels=base_channels,
            dropout=dropout
        ).to(device)

        model, _, _ = train(model, train_loader, val_loader, device,
                            epochs=hparams["epochs"], lr=lr)

        metrics = evaluate(model, val_loader, device)
        print("Val AUC:", metrics["auc"])

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_model = model
            best_config = (base_channels, dropout, lr)

    print("\nBEST CONFIG:", best_config, "AUC:", best_auc)
    return best_model, best_config


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="AB_patches_final_2.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output", default="cnn_out")
    parser.add_argument("--grid-search", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    dataset = PatchDataset(args.data)
    N = len(dataset)

    n_test = int(0.15 * N)
    n_val = int(0.15 * N)
    n_train = N - n_test - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    if args.grid_search:
        hparams = {
            "base_channels": [16, 32, 64],
            "dropout": [0.0, 0.3, 0.5],
            "lr": [1e-3, 5e-4],
            "epochs": 20,
        }

        best_model, best_config = grid_search(
            hparams, dataset, device, train_ds, val_ds, args.batch_size
        )
        model = best_model

    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = PatchCNN(
            in_channels=dataset.X.shape[1],
            base_channels=args.base_channels,
            dropout=args.dropout
        ).to(device)

        model, train_losses, _ = train(
            model, train_loader, val_loader, device, args.epochs, args.lr
        )

        plot_loss(train_losses, f"{args.output}/loss.png")

    # Final evaluation
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    metrics = evaluate(model, test_loader, device)

    print("\n=== FINAL TEST METRICS ===")
    for k, v in metrics.items():
        if k not in ["fpr", "tpr"]:
            print(f"{k}: {v}")

    plot_roc(metrics, f"{args.output}/roc.png")
    np.savez(f"{args.output}/metrics.npz", **metrics)


if __name__ == "__main__":
    main()
