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
    from torch.utils.data import TensorDataset, DataLoader, random_split

    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, roc_auc_score,
        precision_recall_fscore_support
    )
    from sklearn.preprocessing import MinMaxScaler

    import numpy as np
    import pickle, os, random, math
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from sklearn.metrics import ConfusionMatrixDisplay
    import pandas as pd
    from sklearn.metrics import accuracy_score

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

    def relabel_noise_clusters(data, noise_pids, energy_cut):
        """Splits data into clean and reassigned-to-background based on PID and MC energy."""
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


    outdir = 'Classification_AB'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    with open('ABmuons_edep.pkl', 'rb') as f:
        muons = pickle.load(f)
    with open('ABsignal_edep.pkl', 'rb') as f:
        signal = pickle.load(f)
    with open('ABbkg_edep.pkl', 'rb') as f:
        background = pickle.load(f)

    # --- Reassign noise-like clusters to background ---
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

    # --- Feature extraction and split ---
    X_np, y_np = get_features_and_labels(sampled_signal, all_background)


    # --- Scaling X data ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_np)

    X_torch = torch.tensor(X_scaled, dtype=torch.float32)
    y_torch = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_torch, y_torch)

    dataset = TensorDataset(X_torch, y_torch)
    dataset_train, dataset_validate, dataset_test = random_split(dataset, lengths = [0.6,0.2,0.2], generator = torch.Generator().manual_seed(2)) # Split dataset into separate datasets for training & testing

    dloader_train = DataLoader(dataset_train, batch_size = 32, shuffle = True)
    dloader_validate = DataLoader(dataset_validate, batch_size = 32, shuffle = True)

    X_train = torch.vstack([dataset_train[i][0] for i in range(len(dataset_train))])
    y_train = torch.vstack([dataset_train[i][1] for i in range(len(dataset_train))])
    X_valid = torch.vstack([dataset_validate[i][0] for i in range(len(dataset_validate))])
    y_valid = torch.vstack([dataset_validate[i][1] for i in range(len(dataset_validate))])
    X_test = torch.vstack([dataset_test[i][0] for i in range(len(dataset_test))])
    y_test = torch.vstack([dataset_test[i][1] for i in range(len(dataset_test))])


    model = nn.Sequential(
        nn.Linear(6, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1) 
    )

    loss_fcn = nn.BCEWithLogitsLoss() 
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)


    def train_epoch(model, optimizer):
        tot_loss = 0
        valid_loss = 0
        for X_train, y_train in dloader_train:
            y_pred = model(X_train)
            optimizer.zero_grad()
            loss = loss_fcn(y_pred, y_train.reshape(-1,1))
            tot_loss += loss.detach()
            loss.backward()
            optimizer.step()
    
        model.eval()
        for X_valid, y_valid in dloader_validate:
            y_pred_v = model(X_valid)
            vloss = loss_fcn(y_pred_v, y_valid.reshape(-1,1))
            valid_loss += vloss.detach()
        
        return tot_loss/len(dataset_train), valid_loss/len(dataset_validate)

    best_val_acc = 0
    best_roc = 0
    
    # Train 150 epochs
    tloss, vloss = [], []
    for epoch in range(150):
        torch.manual_seed(1)

        train_loss, valid_loss = train_epoch(model, opt)
        tloss.append(train_loss)
        vloss.append(valid_loss)
                

    np.save('AB_losses.npy',np.array([tloss, vloss]))
    torch.save(model, 'AB_NN_Model.pt')

    torch.save({
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test":  X_test,
        "y_test":  y_test
    }, "AB_split_data.pt")

    print("Saved train/valid/test tensors to AB_split_data.pt")


            

if args.evaluate:

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
    from matplotlib.ticker import MultipleLocator


    model = torch.load('AB_NN_Model.pt', weights_only=False) #NEED THE WEIGHTS ONLY PART!!!!
    tloss, vloss = np.load('AB_losses.npy')


    fig, ax = plt.subplots(1,1,figsize = (8,6),dpi = 150)

    ax.plot(tloss, color='black',label='Training loss')
    ax.plot(vloss, color='#D55E00',label='Validation loss')
    ax.set_xlabel('Epoch',fontsize = 16)
    ax.set_ylabel('Binary cross entropy',fontsize = 16)
    ax.set_title('Loss during training',fontsize = 20)
    ax.tick_params(labelsize =12, which = 'both',top=True, right = True, direction='in')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(4))
    ax.grid(color='xkcd:dark blue',alpha = 0.2)
    ax.legend(loc='upper right',fontsize = 12)


    data = torch.load("AB_split_data.pt")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_valid = data["X_valid"]
    y_valid = data["y_valid"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    fig, ax = plt.subplots(1,1,figsize = (8,6),dpi = 150)

    test_accuracy = (torch.Tensor([0 if x < 0.5 else 1 for x in model(X_test)]).reshape(y_test.shape)==y_test).sum()/len(X_test)
    print("Test accuracy = {:.1f}%".format(test_accuracy*100))


    fpr, tpr, thresholds = roc_curve(y_test.detach().numpy(), model(X_test).detach().numpy())
    roc_score = roc_auc_score(y_true = y_test.detach().numpy(),
                            y_score = model(X_test).detach().numpy())

    fig, ax = plt.subplots(1,1,figsize = (6, 4), dpi = 150)
    ax.plot(fpr, tpr, color='#D55E00')
    ax.set_xlabel('False positive rate',fontsize = 20)
    ax.set_ylabel('True positive rate',fontsize = 20)
    ax.set_title('ROC curve, test data',fontsize = 24)
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(MultipleLocator(0.04))
    ax.tick_params(which='both',direction='in',top=True,right=True,labelsize = 16)
    ax.grid(color='xkcd:dark blue',alpha = 0.2)
    print('ROC score = {:.5f}'.format(roc_score))

