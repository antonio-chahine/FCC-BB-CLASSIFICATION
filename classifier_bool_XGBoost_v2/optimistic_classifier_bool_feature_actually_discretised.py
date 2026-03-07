from podio import root_io
import ROOT
import glob
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import functions
import math
import random
from collections import defaultdict

ROOT.gROOT.SetBatch(True)

# === Command-line args ===
parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', help='Process and save muon, signal, and background files')
parser.add_argument('--classify', action='store_true', help='Train and evaluate a classifier')
args = parser.parse_args()

# === Geometry config ===
PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0

# === Process and save data ===
if args.run:
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

                    # snap each hit to the centre of its pixel
                    p.hits = functions.snap_hits_to_pixel_centers(p.hits, pitch_mm=PITCH, radius_mm=RADIUS)

                    # recompute barycenter + cosθ on snapped hits
                    b_x, b_y, b_z = functions.geometric_baricenter(p.hits)
                    cos_theta = functions.cos_theta(b_x, b_y, b_z)

                    # now compute features
                    z_ext = p.z_extent()
                    nrows = p.n_phi_rows(PITCH, RADIUS)

                    # hit_B stays the same
                    hit_B = int(any(functions.in_1B(h) for h in p.hits))

                    cluster_metrics.append((
                        z_ext,
                        nrows,
                        multiplicity,
                        total_edep,
                        mc_energy,
                        cos_theta,
                        b_x, b_y,
                        pid,
                        hit_B
                    ))
        with open(outfile, 'wb') as f:
            pickle.dump(cluster_metrics, f)
        print(f"✅ Saved {label} clusters to {outfile}")



if args.classify:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        precision_recall_fscore_support,
        roc_curve,
        ConfusionMatrixDisplay
    )
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import random
    from functions import relabel_noise_clusters

    def get_features_and_labels(signal_data, background_data, epsilon=1e-6, max_samples=None):
        def transform(row, epsilon=epsilon):
            z, rows, mult, edep, _, cos, _, _, _, hit_B = row
            return (
                math.log(z + epsilon),        # log(z_extent)
                rows,                         # number of φ rows
                mult,                         # multiplicity
                math.log(edep + epsilon),     # log(energy deposition)
                cos,                           # cos(θ)
                hit_B                        # hit in region B (1 if hit, 0 if not)
            )
        
        sig = [(1, transform(row)) for row in signal_data]
        bkg = [(0, transform(row)) for row in background_data]
        data = sig + bkg
        random.shuffle(data)

        if max_samples is not None:
            data = data[:max_samples]

        labels, features = zip(*data)
        return np.array(features), np.array(labels)


    outdir = 'Classification_AB_discretised'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    # === Load data ===
    with open('ABmuons_edep.pkl', 'rb') as f:
        muons = pickle.load(f)
    with open('ABsignal_edep.pkl', 'rb') as f:
        signal = pickle.load(f)
    with open('ABbkg_edep.pkl', 'rb') as f:
        background = pickle.load(f)

    # === Reassign noise-like clusters to background ===
    noise_pids = {11, -11, 13, -211, 22, 211, 2212, -2212}
    energy_cut = 0.01
    clean_muons, reassigned_muons = relabel_noise_clusters(muons, noise_pids, energy_cut)
    clean_signal, reassigned_signal = relabel_noise_clusters(signal, noise_pids, energy_cut)

    all_background = background + reassigned_muons + reassigned_signal
    all_signal = clean_muons + clean_signal
    sampled_signal = random.sample(all_signal, len(all_background))

    print(f"Clean muons: {len(clean_muons)}")
    print(f"Clean signal: {len(clean_signal)}")
    print(f"Reassigned to background: {len(reassigned_muons) + len(reassigned_signal)}")

    # === Feature extraction and split ===
    X, y = get_features_and_labels(sampled_signal, all_background)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # === Sweep thresholds to find one that preserves ≥99% signal
    thresholds = np.linspace(0.0, 1.0, 500)
    tpr_list, fpr_list, f1_list = [], [], []

    for thresh in thresholds:
        y_pred_temp = (y_proba >= thresh).astype(int)
        TP = np.sum((y_pred_temp == 1) & (y_test == 1))
        FP = np.sum((y_pred_temp == 1) & (y_test == 0))
        FN = np.sum((y_pred_temp == 0) & (y_test == 1))
        TN = np.sum((y_pred_temp == 0) & (y_test == 0))

        tpr = TP / (TP + FN) if TP + FN > 0 else 0
        fpr = FP / (FP + TN) if FP + TN > 0 else 0
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_temp, average='binary', zero_division=0)

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        f1_list.append(f1)

    target_tpr = 0.99
    best_thresh, best_fpr = None, 1.0
    for thresh, tpr, fpr in zip(thresholds, tpr_list, fpr_list):
        if tpr >= target_tpr and fpr < best_fpr:
            best_thresh, best_fpr = thresh, fpr

    if best_thresh is not None:
        print(f"Threshold for ≥{target_tpr*100:.1f}% signal retention: {best_thresh:.4f}")
        print(f"Background rejection at that threshold: {1 - best_fpr:.4f}")
    else:
        print(f"No threshold found that satisfies TPR ≥ {target_tpr*100:.1f}%")

    y_pred = (y_proba >= best_thresh).astype(int)

    print("\n=== Final Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Background", "Signal"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC score: %.4f" % roc_auc_score(y_test, y_proba))

    plt.plot(thresholds, tpr_list, label='TPR (Signal Retention)')
    plt.plot(thresholds, fpr_list, label='FPR (Background Acceptance)')
    if best_thresh is not None:
        plt.axvline(best_thresh, color='g', linestyle='--', label=f'TPR ≥ {target_tpr*100:.0f}% @ {best_thresh:.3f}')
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Sweep — TPR, FPR")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "threshold_sweep_metrics.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Final XGBoost Classifier")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "presentation_ROC_curve.png"))
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Background", "Signal"],
        cmap="Blues",
        values_format='d'
    )
    plt.title(f"Confusion Matrix @ Threshold = {best_thresh:.4f}")
    plt.savefig(os.path.join(outdir, "presentation_confusion_matrix.png"))
    plt.close()
    
    functions.plot_feature_importance(
    clf.feature_importances_,
    feature_names=[
        r"$\log(\Delta z)$",
        r"$\varphi$ extent",
        r"multiplicity",
        r"$\log(E_{\mathrm{dep}})$",
        r"$\cos\theta$",
        r"$\mathrm{hit}_B$"
    ],
    outdir="Classification_AB",
    filename="feature_importance",
    sort=True
)
    
    import pickle
    with open("results_classifierB.pkl", "wb") as f:
        pickle.dump((y_test, y_proba), f)


    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import precision_recall_fscore_support
    import numpy as np
    import random
    import math

    def set_all_seeds(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def get_best_threshold_for_target_tpr(y_true, y_proba, target_tpr, n_grid=2000):
        """
        Choose the threshold that achieves TPR >= target_tpr with minimal FPR.
        Returns (best_thresh, best_fpr, best_tpr). If impossible, returns (None, None, None).
        """
        thresholds = np.linspace(0.0, 1.0, n_grid)
        best_thresh, best_fpr, best_tpr = None, 1.0, 0.0

        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba)

        # Precompute positives/negatives counts
        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)
        if P == 0 or N == 0:
            return None, None, None

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))
            TN = np.sum((y_pred == 0) & (y_true == 0))

            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

            if tpr >= target_tpr and fpr < best_fpr:
                best_thresh, best_fpr, best_tpr = float(thresh), float(fpr), float(tpr)

        return best_thresh, best_fpr, best_tpr

    def summarize(values):
        values = np.asarray([v for v in values if v is not None], dtype=float)
        if len(values) == 0:
            return None
        return {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "p16":  float(np.percentile(values, 16)),
            "p50":  float(np.percentile(values, 50)),
            "p84":  float(np.percentile(values, 84)),
            "n":    int(len(values)),
        }

    # ---- choose what you want here ----
    target_tprs = [0.90, 0.99, 0.999]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # increase for tighter uncertainty
    max_samples = None  # or an int if you want to cap events

    results = {t: {"thresh": [], "rej": [], "auc": []} for t in target_tprs}

    for seed in seeds:
        set_all_seeds(seed)

        # === build your dataset with this seed ===
        # (uses your existing variables: all_signal, all_background)
        sampled_signal = random.sample(all_signal, len(all_background))
        X, y = get_features_and_labels(sampled_signal, all_background, max_samples=max_samples)

        # IMPORTANT: stratify to keep class balance stable across seeds
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        clf = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,     # controls xgb RNG
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        # === thresholds for each target TPR ===
        for t in target_tprs:
            best_thresh, best_fpr, best_tpr = get_best_threshold_for_target_tpr(y_test, y_proba, t, n_grid=5000)

            if best_thresh is None:
                results[t]["thresh"].append(None)
                results[t]["rej"].append(None)
                results[t]["auc"].append(float(auc))
                continue

            bkg_rej = 1.0 - best_fpr
            results[t]["thresh"].append(best_thresh)
            results[t]["rej"].append(bkg_rej)
            results[t]["auc"].append(float(auc))

            print(f"[seed={seed}] target TPR={t:.3f}  thresh={best_thresh:.4f}  "
                f"TPR≈{best_tpr:.4f}  bkg_rej={bkg_rej:.4f}  AUC={auc:.4f}")

    print("\n=== Uncertainty across random seeds ===")
    for t in target_tprs:
        s_thr = summarize(results[t]["thresh"])
        s_rej = summarize(results[t]["rej"])
        s_auc = summarize(results[t]["auc"])

        print(f"\nTarget signal retention (TPR) = {100*t:.2f}%")
        if s_thr is None or s_rej is None:
            print("  No valid thresholds found for this target in your runs.")
            continue

        print(f"  threshold: mean={s_thr['mean']:.4f} ± {s_thr['std']:.4f}  "
            f"(p16={s_thr['p16']:.4f}, p50={s_thr['p50']:.4f}, p84={s_thr['p84']:.4f}, n={s_thr['n']})")
        print(f"  bkg rejection: mean={s_rej['mean']:.4f} ± {s_rej['std']:.4f}  "
            f"(p16={s_rej['p16']:.4f}, p50={s_rej['p50']:.4f}, p84={s_rej['p84']:.4f}, n={s_rej['n']})")
        print(f"  AUC: mean={s_auc['mean']:.4f} ± {s_auc['std']:.4f}  "
            f"(p16={s_auc['p16']:.4f}, p50={s_auc['p50']:.4f}, p84={s_auc['p84']:.4f}, n={s_auc['n']})")