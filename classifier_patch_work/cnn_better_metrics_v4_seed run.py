#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score
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
parser.add_argument("--data", default="muons_energycut_nomultiplcity_patches_32size_dphifix.npz")
parser.add_argument("--run", action="store_true", help="Train + evaluate in one go")
parser.add_argument("--seed", type=int, default=2, help="Base seed")
parser.add_argument("--n_seeds", type=int, default=1, help="How many seeds to run: seed, seed+1, ...")
args = parser.parse_args()



class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # no build_classifier needed anymore
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # -> (N, 128, 1, 1)
            nn.Flatten(),             # -> (N, 128)
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
if args.run:
    args.classify = True
    args.evaluate = True

'''
# ======================================================================
# RUN (train + evaluate) over multiple seeds
# ======================================================================
if args.classify or args.evaluate:

    print("Loading dataset...")
    npz = np.load(args.data)
    X = npz["X"].astype(np.float32)      # (N, 2, H, W)
    y = npz["y"].astype(np.float32)      # (N,)

    # NOTE: this normalisation is still computed on all data (fast + consistent).
    # If you want "proper" no-leak normalisation later, we can do that, but keep it simple for now.
    mean = X.mean(axis=(0,2,3), keepdims=True)
    std  = X.std(axis=(0,2,3), keepdims=True) + 1e-6
    X = (X - mean) / std

    X_torch_all = torch.tensor(X, dtype=torch.float32)
    y_torch_all = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset_all = TensorDataset(X_torch_all, y_torch_all)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    aucs = []
    rej_at_99 = []  # background rejection at TPR>=0.99

    base_seed = args.seed

    for k in range(args.n_seeds):
        seed = base_seed + k

        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        outdir = f"CNN_muons_0.005energycut_nomultiplcity_patches_32size_seed{seed}"
        os.makedirs(outdir, exist_ok=True)
        print(f"\n=== Seed {seed} -> {outdir} ===")

        # Split (seeded)
        dataset_train, dataset_validate, dataset_test = random_split(
            dataset_all, [0.6, 0.2, 0.2],
            generator=torch.Generator().manual_seed(seed)
        )

        dloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
        dloader_validate = DataLoader(dataset_validate, batch_size=128, shuffle=False)

        # ---------------------------
        # TRAIN
        # ---------------------------
        model = SimpleCNN().to(device)

        # pos_weight from ALL labels (simple); if you want, change to train-only later.
        N_pos = y_torch_all.sum()
        N_neg = len(y_torch_all) - N_pos
        pos_weight = torch.tensor([(N_neg / (N_pos + 1e-6)).item()], dtype=torch.float32).to(device)

        loss_fcn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)

        tloss, vloss = [], []

        def train_epoch():
            model.train()
            total = 0.0
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
            vtotal = 0.0
            with torch.no_grad():
                for Xv, yv in dloader_validate:
                    Xv = Xv.to(device)
                    yv = yv.to(device)
                    pred_v = model(Xv)
                    vloss_v = loss_fcn(pred_v, yv)
                    vtotal += vloss_v.item()

            return total/len(dloader_train), vtotal/len(dloader_validate)

        best_val = float("inf")
        patience = 30
        wait = 0

        print("Training CNN...")
        for epoch in trange(100, desc=f"Epochs (seed {seed})"):
            train_l, val_l = train_epoch()
            tloss.append(train_l)
            vloss.append(val_l)

            if val_l < best_val:
                best_val = val_l
                wait = 0
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(best_model_state)

        np.save(f"{outdir}/CNN_losses.npy", np.array([tloss, vloss]))
        torch.save(model.state_dict(), f"{outdir}/state_dict.pt")

        # Save split indices so evaluation is consistent
        torch.save({
            "train_idx": dataset_train.indices,
            "val_idx": dataset_validate.indices,
            "test_idx": dataset_test.indices
        }, f"{outdir}/split_idx.pt")

        # ---------------------------
        # EVALUATE
        # ---------------------------
        model = SimpleCNN().cpu()
        model.load_state_dict(torch.load(f"{outdir}/state_dict.pt", map_location="cpu"))
        model.eval()

        test_idx = torch.load(f"{outdir}/split_idx.pt", map_location="cpu")["test_idx"]
        X_test = torch.tensor(X[test_idx], dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32).view(-1)

        with torch.no_grad():
            logits = model(X_test)
            probs = torch.sigmoid(logits).view(-1).numpy()

        labels = y_test.numpy()

        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        aucs.append(roc_auc)

        # Save full ROC arrays (useful later)
        np.savez(f"{outdir}/roc_arrays.npz", fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc)

        # Save poster-friendly threshold numbers
        targets = [0.90, 0.99, 0.999]
        lines = []
        lines.append(f"seed {seed}")
        lines.append(f"ROC_AUC {roc_auc:.6f}")
        for target_tpr in targets:
            valid = np.where(tpr >= target_tpr)[0]
            if len(valid) == 0:
                lines.append(f"TPR>={target_tpr:.3f}: not reached")
                continue
            i = valid[0]
            bkg_rej = 1.0 - fpr[i]
            lines.append(
                f"TPR>={target_tpr:.3f} achieved={tpr[i]:.6f} "
                f"bkg_rej={bkg_rej:.6f} fpr={fpr[i]:.6f} thr={thresholds[i]:.6f}"
            )
            if abs(target_tpr - 0.99) < 1e-12:
                rej_at_99.append(bkg_rej)

        with open(f"{outdir}/roc_thresholds.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

        # ROC plot per seed
        fig, ax = plt.subplots(figsize=(6,5), dpi=150)
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1],'--')
        ax.set_xlabel("False positive rate", fontsize=15)
        ax.set_ylabel("True positive rate", fontsize=15)
        ax.set_title(f"CNN ROC (AUC={roc_auc:.3f})", fontsize=16)
        ax.grid(alpha=0.3)
        plt.savefig(f"{outdir}/CNN_ROC.pdf")
        plt.close()

        print(f"Seed {seed}: AUC={roc_auc:.4f}")

    # ---------------------------
    # SUMMARY over seeds
    # ---------------------------
    aucs = np.array(aucs)
    print("\n==============================")
    print(f"Seeds: {list(range(base_seed, base_seed+args.n_seeds))}")
    print(f"AUC: mean={aucs.mean():.4f}, std={aucs.std():.4f}")
    if len(rej_at_99) > 0:
        rej_at_99 = np.array(rej_at_99)
        print(f"Bkg rejection @ TPR>=0.99: mean={rej_at_99.mean():.4f}, std={rej_at_99.std():.4f}")
    print("==============================")

'''

# ======================================================================
# CLASSIFY
# ======================================================================
if args.classify:

    outdir = "CNN_muons_energycut_nomultiplcity_patches_32size_2"
    os.makedirs(outdir, exist_ok=True)

    print("Loading dataset...")
    npz = np.load(args.data)
    X = npz["X"].astype(np.float32)      # (N, 2, H, W)
    y = npz["y"].astype(np.float32)      # (N,)

    # --- better normalisation ---
    mean = X.mean(axis=(0,2,3), keepdims=True)
    std  = X.std(axis=(0,2,3), keepdims=True) + 1e-6
    X = (X - mean) / std

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


    model = SimpleCNN().to(device)


    # Compute class weights
    N_pos = y_torch.sum()
    N_neg = len(y_torch) - N_pos

    pos_weight = (N_neg / N_pos)
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)

    loss_fcn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

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
    patience = 20
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

    outdir = "CNN_muons_energycut_nomultiplcity_patches_32size"
    os.makedirs(outdir, exist_ok=True)

    # Load model + losses + splits
    model = torch.load(f"{outdir}/CNN_model.pt", map_location="cpu", weights_only=False)
    model = model.cpu()
    model.eval()

    tloss, vloss = np.load(f"{outdir}/CNN_losses.npy")

    data = torch.load(f"{outdir}/CNN_split.pt", map_location="cpu")
    test_set = data["test"]

    X_test = torch.stack([x for x, _ in test_set])
    y_test = torch.stack([y for _, y in test_set]).view(-1)   # (N,)
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
        logits = model(X_test)                 # (N, 1)
        probs  = torch.sigmoid(logits).view(-1).cpu()  # (N,)

    # Accuracy at fixed threshold 0.5
    preds_05 = (probs >= 0.5).float()
    acc_05 = accuracy_score(labels.numpy(), preds_05.numpy())
    print(f"Accuracy @ threshold 0.5 = {acc_05*100:.2f}%")

    # --------------------------------------------------------
    # ROC + AUC
    # --------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(labels.numpy(), probs.numpy())
    roc_auc = roc_auc_score(labels.numpy(), probs.numpy())
    print("ROC AUC =", roc_auc)

    # Best threshold = argmax(Youden J = TPR - FPR)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    print(f"Optimal threshold (Youden J) = {best_thresh:.6f}")

    preds_best = (probs >= best_thresh).float()
    acc_best = accuracy_score(labels.numpy(), preds_best.numpy())
    print(f"Accuracy @ optimal threshold = {acc_best*100:.2f}%")

    # ROC plot
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(fpr, tpr, color="#D55E00")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False positive rate", fontsize=15)
    ax.set_ylabel("True positive rate", fontsize=15)
    ax.set_title("CNN ROC Curve", fontsize=18)
    ax.grid(alpha=0.3)
    plt.savefig(f"{outdir}/CNN_ROC.pdf")
    plt.close()

    # --------------------------------------------------------
    # Efficiency vs Background rejection curve  (TPR vs 1-FPR)
    # --------------------------------------------------------
    bkg_rej = 1.0 - fpr
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(tpr, bkg_rej)
    ax.set_xlabel("Signal efficiency (TPR)", fontsize=15)
    ax.set_ylabel("Background rejection (1 - FPR)", fontsize=15)
    ax.set_title("Efficiency vs Background Rejection", fontsize=18)
    ax.grid(alpha=0.3)
    plt.savefig(f"{outdir}/CNN_eff_vs_bkg_rej.pdf")
    plt.close()

    # --------------------------------------------------------
    # Background rejection at fixed signal efficiencies
    # (90%, 99%, 99.9%)  -- guarantee >= target efficiency
    # --------------------------------------------------------
    targets = [0.90, 0.99, 0.999]
    print("\nBackground rejection at >= fixed signal efficiency:")
    for target_tpr in targets:
        valid = np.where(tpr >= target_tpr)[0]
        if len(valid) == 0:
            print(f"  No operating point reaches {target_tpr*100:.2f}% signal efficiency")
            continue

        idx = valid[0]  # loosest cut that still hits the target (monotonic ROC order)
        sig_eff = tpr[idx]
        bkg_acc = fpr[idx]
        bkg_rej_here = 1.0 - bkg_acc
        thr = thresholds[idx]

        print(
            f"  Sig eff >= {target_tpr*100:6.2f}% | "
            f"achieved {sig_eff*100:6.2f}% | "
            f"bkg rej {bkg_rej_here*100:8.4f}% | "
            f"bkg acc {bkg_acc*100:8.4f}% | "
            f"thr {thr:.6f}"
        )

    # --------------------------------------------------------
    # Precision-Recall curve + PR AUC (Average Precision)
    # --------------------------------------------------------
    precision, recall, pr_thresholds = precision_recall_curve(labels.numpy(), probs.numpy())
    pr_auc = average_precision_score(labels.numpy(), probs.numpy())  # AP = PR AUC summary metric
    print("\nPR AUC (Average Precision) =", pr_auc)

    # PR plot
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall (signal efficiency)", fontsize=15)
    ax.set_ylabel("Precision (signal purity)", fontsize=15)
    ax.set_title(f"Precision-Recall Curve (AP={pr_auc:.4f})", fontsize=16)
    ax.grid(alpha=0.3)
    plt.savefig(f"{outdir}/CNN_PR.pdf")
    plt.close()

    # --------------------------------------------------------
    # Score distributions
    # --------------------------------------------------------
    sig = probs[labels == 1]
    bkg = probs[labels == 0]

    plt.hist(sig.numpy(), bins=50, alpha=0.5, density=True, label="Signal")
    plt.hist(bkg.numpy(), bins=50, alpha=0.5, density=True, label="Background")
    plt.xlabel("Predicted probability", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.title("CNN predicted probabilities", fontsize=18)
    plt.legend()
    plt.savefig(f"{outdir}/CNN_predicted_probabilities.pdf")
    plt.close()
