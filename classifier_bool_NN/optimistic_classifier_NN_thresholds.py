import glob
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', help='Process and save muon, signal, and background files')
parser.add_argument('--classify', action='store_true', help='Train and evaluate a classifier')
parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained classifier')
args = parser.parse_args()



# --- Process and save data ---
if args.run:
    from podio import root_io
    import ROOT
    ROOT.gROOT.SetBatch(True)
    import functions

    # --- Geometry config ---
    PITCH = functions.PITCH_MM
    RADIUS = functions.RADIUS_MM
    LAYER_RADII = [14, 36, 58]
    TARGET_LAYER = 0

    all_configs = {
        'muons': {
            'files': glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root'),
            'outfile': 'ABmuons_edep.pkl'
        },
        'signal': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root'),
            'outfile': 'ABsignal_edep.pkl'
        },
        'background': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'),
            'outfile': 'ABbkg_edep.pkl'
        }
    }

    for label, config in all_configs.items():
        files = config['files']
        outfile = config['outfile']
        cluster_metrics = []
        limit = {
            'muons': 978,
            'signal': 100,
            'background': 1247
        }[label]

        for i, filename in enumerate(files):
            if i >= limit:
                break
            print(f"[{label.upper()}] Processing file {i+1}/{limit}: {filename}")
            reader = root_io.Reader(filename)
            events = reader.get('events')

            for event in events:
                particle_hits = defaultdict(list)

                for hit in event.get('VertexBarrelCollection'):
                    try:
                        if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                            continue
                        if hit.isProducedBySecondary():
                            continue
                        pos = hit.getPosition()
                        mc = hit.getMCParticle()
                        if mc is None:
                            continue
                        trackID = mc.getObjectID().index
                        energy = mc.getEnergy()
                        pid = mc.getPDG()
                        try:
                            edep = hit.getEDep()
                        except AttributeError:
                            edep = 0
                        h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, energy=energy, edep=edep, trackID=trackID)
                        particle_hits[trackID].append((trackID, h, pid))
                    except Exception as e:
                        print(f"Skipping hit due to error: {e}")

                for trackID, hit_group in particle_hits.items():
                    if not hit_group:
                        continue
                    _, _, pid = hit_group[0]
                    ##
                    p = functions.Particle(trackID=trackID)
                    p.pid = pid
                    ##
                    for _, h, _ in hit_group:
                        p.add_hit(h)
                        
                    multiplicity = len(p.hits)
                    if multiplicity == 2:
                        p.hits = functions.merge_cluster_hits(p.hits)
                    total_edep = p.total_energy()
                    b_x, b_y, b_z = functions.geometric_baricenter(p.hits)
                    cos_theta = functions.cos_theta(b_x, b_y, b_z)
                    mc_energy = p.hits[0].energy

                    ### changes
                    z_ext = p.z_extent()

                    if functions.discard_AB(pos): #if cluster passes B, set hit_B to 1
                        hit_B = 1
                    else:
                        hit_B = 0

                    ###


                    nrows = p.n_phi_rows(PITCH, RADIUS)

                    cluster_metrics.append((z_ext, nrows, multiplicity, total_edep, mc_energy, cos_theta, b_x, b_y, pid, hit_B))

        with open(outfile, 'wb') as f:
            pickle.dump(cluster_metrics, f)
        print(f"Saved {label} clusters to {outfile}")



if args.classify:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import (
        roc_curve,
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    import numpy as np
    import pickle, os, random, math
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------
    def get_features_and_labels(signal_data, background_data, epsilon=1e-6, max_samples=None):
        """
        Build the feature matrix X and label vector y.

        Each row from *_edep.pkl has the structure:
        (z_extent, nrows, multiplicity, total_edep, mc_energy,
         cos_theta, b_x, b_y, pid, hit_B)
        """
        def transform(row):
            z, rows, mult, edep, _, cos, _, _, _, hit_B = row
            return (
                math.log(z + epsilon),        # log(z_extent)
                rows,                         # number of φ rows
                mult,                         # multiplicity
                math.log(edep + epsilon),     # log(energy deposition)
                cos,                          # cos(θ)
                hit_B                         # hit in region B (1 if hit, 0 if not)
            )

        sig = [(1, transform(row)) for row in signal_data]
        bkg = [(0, transform(row)) for row in background_data]
        data = sig + bkg
        random.shuffle(data)

        if max_samples is not None:
            data = data[:max_samples]

        labels, features = zip(*data)
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

    def relabel_noise_clusters(data, noise_pids, energy_cut):
        """
        Split clusters into:
          - clean: kept as signal
          - reassigned: treated as background
        based on MC PID and MC energy.
        """
        clean = []
        reassigned = []
        for row in data:
            pid = int(row[8])
            energy = row[4]
            if pid in noise_pids and energy < energy_cut:
                reassigned.append(row)
            else:
                clean.append(row)
        return clean, reassigned

    def scan_thresholds(y_true, y_proba, target_sigeffs, n_steps=1000):
        """
        For each target signal efficiency (TPR), find the threshold that:
          - achieves TPR >= target
          - minimises FPR (maximises background rejection)
        Returns a dict: eff -> (best_threshold, tpr, fpr, bkg_rejection)
        """
        thresholds = np.linspace(0.0, 1.0, n_steps)
        results = {eff: {"thr": None, "tpr": 0.0, "fpr": 1.0} for eff in target_sigeffs}

        # Precompute denominators
        sig_mask = (y_true == 1)
        bkg_mask = (y_true == 0)
        n_sig = np.sum(sig_mask)
        n_bkg = np.sum(bkg_mask)

        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)

            TP = np.sum((y_pred == 1) & sig_mask)
            FP = np.sum((y_pred == 1) & bkg_mask)
            FN = np.sum((y_pred == 0) & sig_mask)
            TN = np.sum((y_pred == 0) & bkg_mask)

            if n_sig == 0 or n_bkg == 0:
                continue

            tpr = TP / (TP + FN + 1e-12)
            fpr = FP / (FP + TN + 1e-12)

            for eff in target_sigeffs:
                rec = results[eff]
                if tpr >= eff and fpr < rec["fpr"]:
                    rec["thr"] = thr
                    rec["tpr"] = tpr
                    rec["fpr"] = fpr

        # Add background rejection and tidy up
        out = {}
        for eff, rec in results.items():
            if rec["thr"] is None:
                out[eff] = None
            else:
                bkg_rej = 1.0 - rec["fpr"]
                out[eff] = (rec["thr"], rec["tpr"], rec["fpr"], bkg_rej)
        return out

    # ------------------------------------------------------------------
    # Load clusters and build dataset
    # ------------------------------------------------------------------
    with open("ABmuons_edep.pkl", "rb") as f:
        muons = pickle.load(f)
    with open("ABsignal_edep.pkl", "rb") as f:
        signal = pickle.load(f)
    with open("ABbkg_edep.pkl", "rb") as f:
        background = pickle.load(f)

    # Reassign low-energy noise-like clusters to background
    noise_pids = {11, -11, 13, -211, 22, 211, 2212, -2212}
    energy_cut = 0.01
    clean_muons, reassigned_muons = relabel_noise_clusters(muons, noise_pids, energy_cut)
    clean_signal, reassigned_signal = relabel_noise_clusters(signal, noise_pids, energy_cut)

    all_background = background + reassigned_muons + reassigned_signal
    all_signal = clean_muons + clean_signal

    # Balance: subsample signal to match total background
    random.seed(42)
    sampled_signal = random.sample(all_signal, len(all_background))

    print(f"Total background clusters: {len(all_background)}")
    print(f"Total clean+reassigned signal clusters: {len(all_signal)}")
    print(f"Using {len(sampled_signal)} signal clusters to balance the dataset.")

    X_np, y_np = get_features_and_labels(sampled_signal, all_background)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_np)

    # Train/valid/test split (fixed so repeats only change init)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_np, test_size=0.3, stratify=y_np, random_state=123
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=123
    )

    # Save the split (for later use if needed)
    torch.save(
        {
            "X_train": torch.tensor(X_train, dtype=torch.float32),
            "y_train": torch.tensor(y_train, dtype=torch.float32),
            "X_valid": torch.tensor(X_valid, dtype=torch.float32),
            "y_valid": torch.tensor(y_valid, dtype=torch.float32),
            "X_test": torch.tensor(X_test, dtype=torch.float32),
            "y_test": torch.tensor(y_test, dtype=torch.float32),
        },
        "AB_split_data.pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def make_loaders(X_train, y_train, X_valid, y_valid, batch_size=512):
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        valid_ds = TensorDataset(
            torch.tensor(X_valid, dtype=torch.float32),
            torch.tensor(y_valid, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=2
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=2
        )
        return train_loader, valid_loader

    def build_model(hidden_dim):
        return nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(device)

    bceloss = nn.BCELoss()

    def train_one_model(hidden_dim, lr, n_epochs, seed):
        # Set seeds for reproducibility of this run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_loader, valid_loader = make_loaders(X_train, y_train, X_valid, y_valid)
        model = build_model(hidden_dim)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best_valid_loss = float("inf")
        for epoch in range(n_epochs):
            # --- training ---
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).view(-1, 1)
                opt.zero_grad()
                out = model(xb)
                loss = bceloss(out, yb)
                loss.backward()
                opt.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            # --- validation ---
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True).view(-1, 1)
                    out = model(xb)
                    loss = bceloss(out, yb)
                    valid_loss += loss.item() * xb.size(0)
            valid_loss /= len(valid_loader.dataset)

        return model

    # ------------------------------------------------------------------
    # Repeated trainings + threshold scan
    # ------------------------------------------------------------------
    target_sigeffs = [0.90, 0.99, 0.999]
    HIDDEN_DIMS = [32, 64]    # small hyperparameter scan
    LRS = [1e-3]              # you can add e.g. 3e-4 here
    N_REPEATS = 5             # per hyperparameter point
    N_EPOCHS = 100

    results = []
    best_run = None

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_np = y_test.astype(int)

    for hidden_dim in HIDDEN_DIMS:
        for lr in LRS:
            for rep in range(N_REPEATS):
                seed = 1000 + rep
                print(f"Training NN: hidden={hidden_dim}, lr={lr}, seed={seed}")
                model = train_one_model(hidden_dim, lr, N_EPOCHS, seed)

                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    y_proba = (
                        model(X_test_t).cpu().numpy().reshape(-1)
                    )
                roc = roc_auc_score(y_test_np, y_proba)
                thr_info = scan_thresholds(y_test_np, y_proba, target_sigeffs)

                run_result = {
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "seed": seed,
                    "roc_auc": roc,
                    "thresholds": thr_info,
                }
                results.append(run_result)

                if (best_run is None) or (roc > best_run["roc_auc"]):
                    best_run = run_result
                    best_y_proba = y_proba  # keep preds for ROC plot

    # Save all numeric results for later analysis
    with open("AB_NN_repeated_results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": results,
                "target_sigeffs": target_sigeffs,
                "y_test": y_test_np,
            },
            f,
        )

    print("\n===== Summary over repeated trainings =====")
    for hidden_dim in HIDDEN_DIMS:
        for lr in LRS:
            subset = [r for r in results if r["hidden_dim"] == hidden_dim and r["lr"] == lr]
            rocs = [r["roc_auc"] for r in subset]
            if not rocs:
                continue
            mean_roc = float(np.mean(rocs))
            std_roc = float(np.std(rocs))
            print(f"hidden={hidden_dim:3d}, lr={lr:.1e}: ROC AUC = {mean_roc:.5f} ± {std_roc:.5f}")
            for eff in target_sigeffs:
                vals = [r["thresholds"][eff] for r in subset if r["thresholds"][eff] is not None]
                if not vals:
                    continue
                bkg_rejs = [v[3] for v in vals]
                mean_rej = float(np.mean(bkg_rejs))
                std_rej = float(np.std(bkg_rejs))
                print(f"    eff={eff*100:6.2f}%: ⟨bkg rej⟩ = {mean_rej:.5f} ± {std_rej:.5f}")

    # ------------------------------------------------------------------
    # ROC curve for the best run
    # ------------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test_np, best_y_proba)
    roc_score = roc_auc_score(y_test_np, best_y_proba)

    outdir = "Classification_AB"
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, label=f"NN (best run) AUC={roc_score:.4f}")
    ax.set_xlabel("False positive rate", fontsize=15)
    ax.set_ylabel("True positive rate", fontsize=15)
    ax.set_title("ROC curve", fontsize=20)
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(MultipleLocator(0.04))
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=16)
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/AB_NN_ROC_curve_best.pdf")
    plt.close()
