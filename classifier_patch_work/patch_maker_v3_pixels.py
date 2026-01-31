#!/usr/bin/env python3

import argparse
import glob
import math
import os
import numpy as np
from collections import defaultdict

from podio import root_io
import ROOT
ROOT.gROOT.SetBatch(True)

import functions

# ---- choose pixel size  ----
functions.PITCH_MM = 0.003125


PITCH_MM  = functions.PITCH_MM
RADIUS_MM = functions.RADIUS_MM
MAX_Z     = functions.max_z

# recompute bins so they match the chosen PITCH_MM
N_PHI_BINS = int((2 * math.pi * RADIUS_MM) / PITCH_MM)
N_Z_BINS   = int((2 * MAX_Z) / PITCH_MM)


LAYER_RADII   = [14, 36, 58]
TARGET_LAYER  = 0
R_BOUNDARY_AB = 13.919

REF_WIDTH_MM = 32 * 0.025   # 0.8 mm (the original patch physical width)
DEFAULT_PATCH_SIZE = int(round(REF_WIDTH_MM / PITCH_MM))
if DEFAULT_PATCH_SIZE % 2 == 1:
    DEFAULT_PATCH_SIZE += 1
DEFAULT_PATCH_SIZE = max(4, DEFAULT_PATCH_SIZE)

# Noise definition for reassignment
noise_pids = {11, -11, 22}
ENERGY_CUT = 0.01  # adjustable threshold (0.01 standard)

# ================================
# Input samples
# ================================
SAMPLES = {
    'muons': {
        'files': glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root'),
        'label': 1
        },
    "signal": {
        "files": glob.glob("/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root"),
        "label": 1
    },
    "background": {
        "files": glob.glob("/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root"),
        "label": 0
    },
}

EVENT_LIMITS = {"muons":978, "signal":100, "background":1247}

# ================================
# Helpers
# ================================
def get_layer_AB(hit):
    r = math.hypot(hit.x, hit.y)
    return 0 if r < R_BOUNDARY_AB else 1


def classify_patch(patch_hits, source_label):
    """
    Reassignment rule:
    - background files stay 0
    - signal/muon files reassigned → 0 only if:
        all hits are noise-PID AND low-energy
    """
    if source_label == 0:
        return 0   # ALWAYS background

    # Check noise conditions
    all_noise_pid = all(h.pid in noise_pids for h in patch_hits)
    all_low_mc_energy = all(h.energy < ENERGY_CUT for h in patch_hits)

    if all_noise_pid and all_low_mc_energy:
        return 0

    return 1


def make_patch(event_hits, bx, by, bz, patch_size, source_label):
    patch_half = patch_size // 2

    # jitter
    z_c, phi_c = functions.get_grid_indices(bx, by, bz)
    REF_JITTER_MM = 4 * 0.025  # = 0.1 mm (same as old ±4 pixels at 0.025)
    jitter_pix = int(round(REF_JITTER_MM / PITCH_MM))
    z_c   += np.random.randint(-jitter_pix, jitter_pix + 1)
    phi_c += np.random.randint(-jitter_pix, jitter_pix + 1)


    # clamp
    z_c   = int(np.clip(z_c,   patch_half, N_Z_BINS  - patch_half - 1))
    phi_c = int(np.clip(phi_c, patch_half, N_PHI_BINS - patch_half - 1))

    patch = np.zeros((2, patch_size, patch_size), dtype=np.float32)
    patch_hits = []

    for h in event_hits:
        z_idx, phi_idx = functions.get_grid_indices(h.x, h.y, h.z)

        phi_idx %= N_PHI_BINS
        z_idx = max(0, min(N_Z_BINS - 1, z_idx))
        dz = z_idx - z_c
        
        dphi = ((phi_idx - phi_c + N_PHI_BINS//2) % N_PHI_BINS) - N_PHI_BINS//2
        
        if abs(dz) >= patch_half or abs(dphi) >= patch_half:
            continue

        iz = dz + patch_half
        iphi = dphi + patch_half

        layer = get_layer_AB(h)
        #Energy deposition
        patch[layer, iz, iphi] += h.edep
        '''
        #Hit multiplicity
        patch[layer + 2, iz, iphi] += 1
        '''
        patch_hits.append(h)

    if not patch_hits:
        return None, None

    label = classify_patch(patch_hits, source_label)
    return patch, label


# ================================
# Process a ROOT file
# ================================
def process_file(filename, sample_label, sample_name, patch_size, max_events):
    print(f"[{sample_name.upper()}] {filename}")

    reader = root_io.Reader(filename)
    events = reader.get("events")

    patches = []
    labels = []

    event_count = 0

    for event in events:
        if max_events and event_count >= max_events:
            break

        event_hits = []
        particles = defaultdict(list)

        # Load hits
        for hit in event.get("VertexBarrelCollection"):
            try:
                if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                    continue

                pos = hit.getPosition()
                mc  = hit.getMCParticle()

                # Handle MC = None properly
                if mc is None:
                    pid = None
                    trackID = -1
                    energy = 999.0 #doesn't affect CNN because only edep (pixel energy) is used. Need to cut out background like cuts
                else:
                    pid = mc.getPDG()
                    trackID = mc.getObjectID().index
                    energy = mc.getEnergy()

                try:
                    edep = hit.getEDep()
                except:
                    edep = 0.0

                h = functions.Hit(
                    x=pos.x, y=pos.y, z=pos.z,
                    energy=energy,
                    edep=edep,
                    trackID=trackID
                )
                h.pid = pid

                event_hits.append(h)
                particles[trackID].append(h)

            except Exception as e:
                print("Hit error:", e)

        # Build patch for each particle
        for trackID, hitlist in particles.items():

            # optional merging
            if len(hitlist) == 2:
                hitlist = functions.merge_cluster_hits(hitlist)

            bx, by, bz = functions.geometric_baricenter(hitlist)

            patch, label = make_patch(
                event_hits, bx, by, bz, patch_size, sample_label
            )
            if patch is None:
                continue

            patches.append(patch)
            labels.append(label)

        event_count += 1

    return patches, labels


# ================================
# Main
# ================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="muons_energycut_nomultiplcity_patches_32size_0.003125pixel.npz")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    all_patches = []
    all_labels = []

    for sname, cfg in SAMPLES.items():
        print(f"\n=== {sname.upper()} ===")

        files = cfg["files"]
        label = cfg["label"]
        max_events = EVENT_LIMITS[sname]

        if args.max_files is not None:
            files = files[:args.max_files]

        BAD_FILES = {
            "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/240499.root",
            "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/444948.root",
            "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/839982.root"

        }

        for f in files:
            # Skip known-bad files
            if f in BAD_FILES:
                print(f"Skipping known corrupted file: {f}")
                continue
            p, l = process_file(f, label, sname, args.patch_size, max_events)
            all_patches.extend(p)
            all_labels.extend(l)

    X = np.stack(all_patches)
    y = np.array(all_labels)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, X=X, y=y)

    print("Saved:", args.out)
    print("Total patches:", len(X))
    print("Class balance:", np.unique(y, return_counts=True))


if __name__ == "__main__":
    main()
