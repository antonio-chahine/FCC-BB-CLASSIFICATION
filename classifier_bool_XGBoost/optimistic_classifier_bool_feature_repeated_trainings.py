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

                    ###changes

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
        def transform(row):
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


    outdir = 'Classification_AB'
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

    # ============================
    # Repeated decision-tree training
    # ============================
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, roc_curve
    import numpy as np

    N_REPEATS = 10
    target_sigeffs = [0.90, 0.99, 0.999]

    def scan_thresholds(y_true, y_proba, effs):
        out = {}
        thresholds = np.linspace(0, 1, 2000)
        sig = (y_true == 1)
        bkg = (y_true == 0)

        for eff in effs:
            best = None
            for thr in thresholds:
                yp = (y_proba >= thr).astype(int)

                TP = np.sum(yp * sig)
                FN = np.sum((1 - yp) * sig)
                FP = np.sum(yp * bkg)
                TN = np.sum((1 - yp) * bkg)

                TPR = TP / (TP + FN + 1e-12)
                FPR = FP / (FP + TN + 1e-12)

                if TPR >= eff:
                    if (best is None) or (FPR < best[2]):
                        best = (thr, TPR, FPR, 1 - FPR)  # threshold, TPR, FPR, bkg_rej
            out[eff] = best
        return out

    results = []

    for rep in range(N_REPEATS):
        print(f"\n=== XGBoost training repeat {rep+1}/{N_REPEATS} ===")

        clf = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=0.5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=1000 + rep,
        )

        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        thr_info = scan_thresholds(y_test, y_proba, target_sigeffs)

        results.append({
            "auc": auc,
            "thresholds": thr_info,
            "seed": 1000 + rep
        })

    # ============================
    # Print summary statistics
    # ============================
    aucs = [r["auc"] for r in results]
    print("\n====== DECISION TREE SUMMARY ======")
    print(f"AUC mean = {np.mean(aucs):.6f}   std = {np.std(aucs):.6f}")

    for eff in target_sigeffs:
        rejs = []
        for r in results:
            data = r["thresholds"][eff]
            if data is not None:
                rejs.append(data[3])  # background rejection
        if rejs:
            print(f"Bkg rejection @ {eff*100:.1f}% signal: {np.mean(rejs):.6f} ± {np.std(rejs):.6f}")
        else:
            print(f"No thresholds found for eff={eff}")

    # Save results for later comparison (NN vs DT)
    with open("results_decision_tree_repeated.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nSaved repeated training results → results_decision_tree_repeated.pkl")
