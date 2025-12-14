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

'''
things to change: --data input file name
outdir in classify and evaluate sections (need to be the same folder)
'''

parser = argparse.ArgumentParser()
parser.add_argument("--classify", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--data", default="muons_energyv2_multiplicity_patches.npz")
args = parser.parse_args()

# ======================================================================
# CLASSIFY
# ======================================================================
if args.classify:

    outdir = "CNN_muons_energyv2_multiplicity_patches_2"
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

    dloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dloader_validate = DataLoader(dataset_validate, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    

    # ======================================================================
    # CNN MODEL (same style as notes: Sequential + simple)
    # ======================================================================
    model = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.1),

        nn.Conv2d(16, (2*16), kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.1),

        nn.Flatten(),
        nn.Linear((2*16) * 8 * 8, 1), 
    ).to(device)


    # Compute class weights
    N_pos = y_torch.sum()
    N_neg = len(y_torch) - N_pos

    pos_weight = (N_neg / N_pos)
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)

    loss_fcn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

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


    best_val = float("inf")
    patience = 30
    wait = 0

    print("Training CNN...")
    for epoch in trange(100, desc="Epochs"):
        train_l, val_l = train_epoch()
        tloss.append(train_l)
        vloss.append(val_l)

        # Early stopping
        if val_l < best_val:
            best_val = val_l
            wait = 0
            best_model_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # restore best model
    model.load_state_dict(best_model_state)

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
    
    outdir = "CNN_muons_energyv2_multiplicity_patches_2"
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
    labels = y_test.cpu()

    # --------------------------------------------------------
    # Loss curves
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu()

    # Accuracy at fixed threshold 0.5
    preds_05 = (probs >= 0.5).float()
    acc_05 = accuracy_score(labels, preds_05)
    print(f"Accuracy @ threshold 0.5 = {acc_05*100:.2f}%")

    # --------------------------------------------------------
    # ROC + AUC
    # --------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(labels.numpy(), probs.numpy())
    auc = roc_auc_score(labels.numpy(), probs.numpy())
    print("ROC AUC =", auc)

    # Best threshold = argmax(Youden J = TPR - FPR)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    print(f"Optimal threshold = {best_thresh:.5f}")

    # Accuracy at optimal threshold
    preds_best = (probs >= best_thresh).float()
    acc_best = accuracy_score(labels, preds_best)
    print(f"Accuracy @ optimal threshold = {acc_best*100:.2f}%")

    # --------------------------------------------------------
    # ROC plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(fpr, tpr, color="#D55E00")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False positive rate", fontsize=15)
    ax.set_ylabel("True positive rate", fontsize=15)
    ax.set_title("CNN ROC Curve", fontsize=18)
    ax.grid(alpha=0.3)
    plt.savefig(f"{outdir}/CNN_ROC.pdf")
    plt.close()

