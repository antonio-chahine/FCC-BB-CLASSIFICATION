#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("--classify", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--data", default="AB_patches_final_2.npz")
args = parser.parse_args()

# ======================================================================
# CLASSIFY
# ======================================================================
if args.classify:

    outdir = "CNN_AB"
    os.makedirs(outdir, exist_ok=True)

    print("Loading dataset...")
    npz = np.load(args.data)
    X = npz["X"].astype(np.float32)      # (N, 2, H, W)
    y = npz["y"].astype(np.float32)      # (N,)

    # --- simple normalisation ---
    X /= np.max(X)

    # Convert to torch
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_torch, y_torch)

    dataset_train, dataset_validate, dataset_test = random_split(
        dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(2)
    )

    dloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
    dloader_validate = DataLoader(dataset_validate, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    

    # ======================================================================
    # CNN MODEL (same style as notes: Sequential + simple)
    # ======================================================================
    model = nn.Sequential(
        nn.Conv2d(2, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 1),   # for 32x32 input
        nn.Sigmoid()
    ).to(device)

    loss_fcn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ======================================================================
    # TRAINING LOOP (same simple style)
    # ======================================================================
    tloss, vloss = [], []

    def train_epoch():
        model.train()
        total = 0

        for Xb, yb in dloader_train:
            Xb = Xb.to(device)
            yb = yb.to(device)

            pred = model(Xb)
            loss = loss_fcn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        # Validation
        model.eval()
        vtotal = 0
        with torch.no_grad():
            for Xv, yv in dloader_validate:
                Xv = Xv.to(device)
                yv = yv.to(device)
                pred_v = model(Xv)
                vloss = loss_fcn(pred_v, yv)
                vtotal += vloss.item()

        return total/len(dloader_train), vtotal/len(dloader_validate)

    print("Training CNN...")
    for epoch in trange(200, desc="Epochs"):
        train_l, val_l = train_epoch()
        tloss.append(train_l)
        vloss.append(val_l)

    np.save(f"{outdir}/CNN_losses.npy", np.array([tloss, vloss]))
    torch.save(model, f"{outdir}/CNN_model.pt")

    # Save train/val/test for evaluation
    torch.save({
        "train": dataset_train,
        "validate": dataset_validate,
        "test": dataset_test
    }, f"{outdir}/CNN_split.pt")

    print("Training complete.")


# ======================================================================
# EVALUATE
# ======================================================================
if args.evaluate:

    outdir = "CNN_AB"
    os.makedirs(outdir, exist_ok=True)

    # Load model + losses + splits
    model = torch.load(f"{outdir}/CNN_model.pt", weights_only=False)
    model = model.cpu()
    model.eval()

    tloss, vloss = np.load(f"{outdir}/CNN_losses.npy")

    data = torch.load(f"{outdir}/CNN_split.pt")
    test_set = data["test"]

    X_test = torch.stack([x for x, _ in test_set])
    y_test = torch.stack([y for _, y in test_set])

    # Loss curves
    fig, ax = plt.subplots(figsize=(8,6), dpi=150)
    ax.plot(tloss, label="Training loss", color="black")
    ax.plot(vloss, label="Validation loss", color="#D55E00")
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Binary cross entropy", fontsize=16)
    ax.set_title("CNN training loss", fontsize=20)
    ax.tick_params(labelsize=12, which="both", top=True, right=True, direction="in")
    ax.grid(color="xkcd:dark blue", alpha=0.2)
    ax.legend()
    plt.savefig(f"{outdir}/CNN_loss.pdf")
    plt.close()

    # Predictions
    with torch.no_grad():
        preds = model(X_test).cpu()
        labels = y_test.cpu()
        preds_bin = (preds >= 0.5).float()

    acc = accuracy_score(labels, preds_bin)
    print(f"Test accuracy = {acc*100:.1f}%")

    # ROC curve
    fpr, tpr, _ = roc_curve(labels.numpy(), preds.numpy())
    auc = roc_auc_score(labels.numpy(), preds.numpy())
    print("ROC AUC =", auc)

    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(fpr, tpr, color="#D55E00")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False positive rate", fontsize=15)
    ax.set_ylabel("True positive rate", fontsize=15)
    ax.set_title("CNN ROC Curve", fontsize=18)
    ax.grid(alpha=0.3)
    plt.savefig(f"{outdir}/CNN_ROC.pdf")
    plt.close()

