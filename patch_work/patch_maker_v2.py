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

# ================================
# Geometry
# ================================
PITCH_MM   = functions.PITCH_MM
RADIUS_MM  = functions.RADIUS_MM
MAX_Z      = functions.max_z
N_PHI_BINS = functions.n_phi_bins
N_Z_BINS   = functions.n_z_bins

LAYER_RADII   = [14, 36, 58]
TARGET_LAYER  = 0
R_BOUNDARY_AB = 13.919

DEFAULT_PATCH_SIZE = 32

# Noise definition for reassignment
noise_pids = {11, -11, 13, -13, -211, 22, 211, 2212, -2212}
ENERGY_CUT = 0.0000   # adjustable threshold

# ================================
# Input samples
# ================================
SAMPLES = {
    "signal": {
        "files": glob.glob("/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root"),
        "label": 1
    },
    "background": {
        "files": glob.glob("/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root"),
        "label": 0
    },
}

EVENT_LIMITS = {"signal":100, "background":1247}

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
    all_noise_pid = all((h.pid in noise_pids or h.pid is None) for h in patch_hits)
    all_low_energy = all(h.edep < ENERGY_CUT for h in patch_hits)

    if all_noise_pid and all_low_energy:
        return 0

    return 1


def make_patch(event_hits, bx, by, bz, patch_size, source_label):
    patch_half = patch_size // 2

    # jitter
    z_c, phi_c = functions.get_grid_indices(bx, by, bz)
    z_c += np.random.randint(-4, 5)
    phi_c += np.random.randint(-4, 5)

    # clamp
    z_c   = int(np.clip(z_c,   patch_half, N_Z_BINS  - patch_half - 1))
    phi_c = int(np.clip(phi_c, patch_half, N_PHI_BINS - patch_half - 1))

    patch = np.zeros((2, patch_size, patch_size), dtype=np.float32)
    patch_hits = []

    for h in event_hits:
        z_idx, phi_idx = functions.get_grid_indices(h.x, h.y, h.z)

        dz = z_idx - z_c
        dphi = phi_idx - phi_c

        if abs(dz) >= patch_half or abs(dphi) >= patch_half:
            continue

        iz = dz + patch_half
        iphi = dphi + patch_half

        layer = get_layer_AB(h)
        patch[layer, iz, iphi] += h.edep
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
                    energy = 0.0
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
    parser.add_argument("--out", type=str, default="AB_patches_final_2.npz")
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
                print(f"❌ Skipping known corrupted file: {f}")
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
