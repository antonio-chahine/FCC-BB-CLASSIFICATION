#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
from tqdm import trange

# ============================================================
# One-time conversion helper: .npz -> X.npy / y.npy (memmapable)
# ============================================================
from tqdm import tqdm
import numpy as np
import os

def convert_npz_to_npy(npz_path: str, x_out: str, y_out: str, chunk: int = 1024):
    os.makedirs(os.path.dirname(x_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(y_out) or ".", exist_ok=True)

    # 👇 key change
    npz = np.load(npz_path, mmap_mode="r")

    X = npz["X"]
    y = npz["y"]

    X_out = np.lib.format.open_memmap(
        x_out, mode="w+", dtype=np.float32, shape=X.shape
    )

    for i in tqdm(range(0, X.shape[0], chunk), desc="Converting X", unit="rows"):
        X_out[i:i+chunk] = X[i:i+chunk].astype(np.float32, copy=False)

    del X_out

    y_out_mem = np.lib.format.open_memmap(
        y_out, mode="w+", dtype=np.float32, shape=y.shape
    )

    for i in tqdm(range(0, y.shape[0], chunk), desc="Converting y", unit="rows"):
        y_out_mem[i:i+chunk] = y[i:i+chunk].astype(np.float32, copy=False)

    del y_out_mem

    print("Conversion finished.")

# ============================================================
# Streaming mean/std over memmap (no full RAM load)
# ============================================================
def compute_channel_mean_std_memmap(x_mmap: np.memmap, batch: int = 4096):
    """
    x_mmap: (N, C, H, W) float32 memmap (or compatible)
    returns mean,std shaped (1,C,1,1) float32
    """
    N, C, H, W = x_mmap.shape
    sum_c = np.zeros((C,), dtype=np.float64)
    sumsq_c = np.zeros((C,), dtype=np.float64)
    count = 0

    for i in range(0, N, batch):
        xb = x_mmap[i:i + batch].astype(np.float32, copy=False)
        sum_c += xb.sum(axis=(0, 2, 3))
        sumsq_c += (xb * xb).sum(axis=(0, 2, 3))
        count += xb.shape[0] * H * W

    mean = (sum_c / count).astype(np.float32).reshape(1, C, 1, 1)
    var = (sumsq_c / count) - (mean.reshape(C) ** 2)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32).reshape(1, C, 1, 1)
    return mean, std

# ============================================================
# Memmap Dataset
# ============================================================
class MemmapPatches(torch.utils.data.Dataset):
    def __init__(self, x_path: str, y_path: str, mean: np.ndarray = None, std: np.ndarray = None):
        self.X = np.load(x_path, mmap_mode="r")  # (N,C,H,W)
        self.y = np.load(y_path, mmap_mode="r")  # (N,)
        self.mean = mean
        self.std = std

        assert self.X.ndim == 4, f"Expected X (N,C,H,W), got {self.X.shape}"
        assert self.y.ndim == 1, f"Expected y (N,), got {self.y.shape}"
        assert self.X.shape[0] == self.y.shape[0], "X and y length mismatch"

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32, copy=False)
        y = np.float32(self.y[idx])

        if self.mean is not None and self.std is not None:
            x = (x - self.mean[0]) / self.std[0]


        x = torch.from_numpy(x)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y

# ============================================================
# Model
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
        )
        self.classifier = None  # built later

    def build_classifier(self, input_shape, device):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape, device=device)
            x = self.features(x)
            n_flat = x.numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 1),
        ).to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
# Split helpers
# ============================================================
def split_indices(N: int, seed: int = 2, frac_train=0.6, frac_val=0.2):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    n_train = int(frac_train * N)
    n_val = int(frac_val * N)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    return idx_train, idx_val, idx_test

def compute_pos_weight_from_memmap(y_mmap: np.memmap):
    N = y_mmap.shape[0]
    N_pos = float(y_mmap[:].sum())  # y is 1D; OK to read
    N_neg = float(N - N_pos)
    if N_pos == 0:
        raise ValueError("No positive examples found (N_pos=0).")
    return N_neg / N_pos

# ============================================================
# Train
# ============================================================
def train_model(x_path, y_path, outdir, batch_size=128, epochs=100, patience=30, seed=2, num_workers=2):
    os.makedirs(outdir, exist_ok=True)

    X_mmap = np.load(x_path, mmap_mode="r")
    y_mmap = np.load(y_path, mmap_mode="r")

    print("Computing mean/std (streaming, no full RAM load)...")
    mean, std = compute_channel_mean_std_memmap(X_mmap, batch=4096)
    std = std + 1e-6

    dataset = MemmapPatches(x_path, y_path, mean=mean, std=std)
    idx_train, idx_val, idx_test = split_indices(len(dataset), seed=seed)

    train_set = Subset(dataset, idx_train)
    val_set = Subset(dataset, idx_val)

    dloader_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    dloader_val = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model = SimpleCNN().to(device)
    H, W = X_mmap.shape[2], X_mmap.shape[3]
    model.build_classifier((2, H, W), device)

    pos_weight_val = compute_pos_weight_from_memmap(y_mmap)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    loss_fcn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    tloss, vloss = [], []

    def train_epoch():
        model.train()
        total = 0.0
        for Xb, yb in dloader_train:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(Xb)
            loss = loss_fcn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item())

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for Xv, yv in dloader_val:
                Xv = Xv.to(device, non_blocking=True)
                yv = yv.to(device, non_blocking=True)
                pred_v = model(Xv)
                loss_v = loss_fcn(pred_v, yv)
                vtotal += float(loss_v.item())

        return total / max(1, len(dloader_train)), vtotal / max(1, len(dloader_val))

    print("Training CNN...")
    best_val = float("inf")
    wait = 0
    best_model_state = None

    for _ in trange(epochs, desc="Epochs"):
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

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    np.save(os.path.join(outdir, "CNN_losses.npy"), np.array([tloss, vloss], dtype=np.float32))
    torch.save(model, os.path.join(outdir, "CNN_model.pt"))
    torch.save({
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "mean": mean,
        "std": std,
        "x_path": x_path,
        "y_path": y_path
    }, os.path.join(outdir, "CNN_split_and_norm.pt"))

    print("Training complete.")
    return outdir

# ============================================================
# Evaluate (plots go to plotdir, model+metadata stay in outdir)
# ============================================================
def evaluate_model(outdir, plotdir, batch_size=256, num_workers=2):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plotdir, exist_ok=True)

    model = torch.load(os.path.join(outdir, "CNN_model.pt"), map_location="cpu", weights_only=False)
    model = model.cpu()
    model.eval()

    tloss, vloss = np.load(os.path.join(outdir, "CNN_losses.npy"))

    info = torch.load(os.path.join(outdir, "CNN_split_and_norm.pt"), map_location="cpu")
    idx_test = info["idx_test"]
    mean = info["mean"]
    std = info["std"]
    x_path = info["x_path"]
    y_path = info["y_path"]

    dataset = MemmapPatches(x_path, y_path, mean=mean, std=std)
    test_set = Subset(dataset, idx_test)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(tloss, label="Training loss", color="black")
    ax.plot(vloss, label="Validation loss", color="#D55E00")
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Binary cross entropy", fontsize=16)
    ax.set_title("CNN training loss", fontsize=20)
    ax.tick_params(labelsize=12, which="both", top=True, right=True, direction="in")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.savefig(os.path.join(plotdir, "CNN_loss.pdf"))
    plt.close()

    # Predictions
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits = model(Xb)  # CPU
            probs = torch.sigmoid(logits).view(-1)
            all_probs.append(probs)
            all_labels.append(yb.view(-1))

    probs = torch.cat(all_probs).cpu()
    labels = torch.cat(all_labels).cpu()

    # Accuracy @ 0.5
    preds_05 = (probs >= 0.5).float()
    acc_05 = accuracy_score(labels.numpy(), preds_05.numpy())
    print(f"Accuracy @ threshold 0.5 = {acc_05*100:.2f}%")

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(labels.numpy(), probs.numpy())
    roc_auc = roc_auc_score(labels.numpy(), probs.numpy())
    print("ROC AUC =", roc_auc)

    # Best threshold (Youden J)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    print(f"Optimal threshold (Youden J) = {best_thresh:.6f}")

    preds_best = (probs >= best_thresh).float()
    acc_best = accuracy_score(labels.numpy(), preds_best.numpy())
    print(f"Accuracy @ optimal threshold = {acc_best*100:.2f}%")

    # ROC plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, color="#D55E00")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False positive rate", fontsize=15)
    ax.set_ylabel("True positive rate", fontsize=15)
    ax.set_title("CNN ROC Curve", fontsize=18)
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(plotdir, "CNN_ROC.pdf"))
    plt.close()

    # Efficiency vs background rejection
    bkg_rej = 1.0 - fpr
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(tpr, bkg_rej)
    ax.set_xlabel("Signal efficiency (TPR)", fontsize=15)
    ax.set_ylabel("Background rejection (1 - FPR)", fontsize=15)
    ax.set_title("Efficiency vs Background Rejection", fontsize=18)
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(plotdir, "CNN_eff_vs_bkg_rej.pdf"))
    plt.close()

    # Background rejection at fixed signal efficiencies
    targets = [0.90, 0.99, 0.999]
    print("\nBackground rejection at >= fixed signal efficiency:")
    for target_tpr in targets:
        valid = np.where(tpr >= target_tpr)[0]
        if len(valid) == 0:
            print(f"  No operating point reaches {target_tpr*100:.2f}% signal efficiency")
            continue
        idx = valid[0]
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

    # Precision-Recall + AP
    precision, recall, _ = precision_recall_curve(labels.numpy(), probs.numpy())
    pr_auc = average_precision_score(labels.numpy(), probs.numpy())
    print("\nPR AUC (Average Precision) =", pr_auc)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall (signal efficiency)", fontsize=15)
    ax.set_ylabel("Precision (signal purity)", fontsize=15)
    ax.set_title(f"Precision-Recall Curve (AP={pr_auc:.4f})", fontsize=16)
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(plotdir, "CNN_PR.pdf"))
    plt.close()

    # Score distributions
    sig = probs[labels == 1]
    bkg = probs[labels == 0]
    plt.hist(sig.numpy(), bins=50, alpha=0.5, density=True, label="Signal")
    plt.hist(bkg.numpy(), bins=50, alpha=0.5, density=True, label="Background")
    plt.xlabel("Predicted probability", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.title("CNN predicted probabilities", fontsize=18)
    plt.legend()
    plt.savefig(os.path.join(plotdir, "CNN_predicted_probabilities.pdf"))
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    # Input options:
    #   --npz <file.npz> will convert to X.npy/y.npy at --x/--y paths
    #   --x / --y can point to Ceph (recommended)
    parser.add_argument("--npz", type=str, default=None, help="Optional input .npz with X,y (will be converted)")

    parser.add_argument("--x", type=str, default="X.npy", help="Memmapable X .npy (N,2,H,W)")
    parser.add_argument("--y", type=str, default="y.npy", help="Memmapable y .npy (N,)")

    # Big outputs (model, split, losses) - can be Ceph
    parser.add_argument("--outdir", type=str, default="CNN_out_memmap")

    # Plots directory (can be local)
    parser.add_argument("--plotdir", type=str, default="plots")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.npz is not None:
        convert_npz_to_npy(args.npz, x_out=args.x, y_out=args.y)

    if args.classify:
        train_model(
            x_path=args.x,
            y_path=args.y,
            outdir=args.outdir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            num_workers=args.num_workers
        )

    if args.evaluate:
        evaluate_model(
            outdir=args.outdir,
            plotdir=args.plotdir,
            batch_size=max(256, args.batch_size),
            num_workers=args.num_workers
        )

if __name__ == "__main__":
    main()
