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


# ============================================================
# Geometry
# ============================================================
PITCH_MM   = functions.PITCH_MM
RADIUS_MM  = functions.RADIUS_MM
MAX_Z      = functions.max_z
N_PHI_BINS = functions.n_phi_bins
N_Z_BINS   = functions.n_z_bins

LAYER_RADII   = [14, 36, 58]
TARGET_LAYER  = 0          # layer 1 (A+B)
R_BOUNDARY_AB = 14.0       # inner/outer split


# ============================================================
# PID definitions
# ============================================================
# signal (muons + electrons)
signal_pids = {11, -11, 13, -13}

# noise PIDs — if *all* hits in patch are these: DROP patch
noise_pids = {11, -11, 13, -211, 22, 211, 2212, -2212}


# ============================================================
# Optional hit energy threshold (pixel activation)
# ============================================================
EDEP_THRESHOLD = 0.0    # change to e.g. 0.0005 if desired


# ============================================================
# Patch size, sample limits
# ============================================================
DEFAULT_PATCH_SIZE = 32

EVENT_LIMITS = {
    "muons":      978,
    "signal":     100,
    "background": 1247,
}

# ROOT file inputs
SAMPLES = {
    "muons": {
        "files": glob.glob(
            "/ceph/submit/data/user/j/jaeyserm/fccee/"
            "beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root"
        )
    },
    "signal": {
        "files": glob.glob(
            "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/"
            "CLD_wz3p6_ee_qq_ecm91p2/*.root"
        )
    },
    "background": {
        "files": glob.glob(
            "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/"
            "CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root"
        )
    },
}


# ============================================================
# Helpers
# ============================================================
def get_layer_AB(hit):
    r = math.sqrt(hit.x**2 + hit.y**2)
    return 0 if r < R_BOUNDARY_AB else 1


def barycenter_indices(bx, by, bz):
    return functions.get_grid_indices(bx, by, bz)


def is_center_valid(z_idx, phi_idx, patch_half):
    if z_idx < patch_half or z_idx >= (N_Z_BINS - patch_half):
        return False
    if phi_idx < patch_half or phi_idx >= (N_PHI_BINS - patch_half):
        return False
    return True


# ============================================================
# Build patch
# ============================================================
def make_patch_for_particle(event_hits, bx, by, bz, patch_size):
    patch_half = patch_size // 2
    z_c, phi_c = barycenter_indices(bx, by, bz)

    if not is_center_valid(z_c, phi_c, patch_half):
        return None, None

    patch = np.zeros((2, patch_size, patch_size), dtype=np.float32)

    hits_inside_patch = []
    patch_has_signal = False

    for h in event_hits:
        z_idx, phi_idx = functions.get_grid_indices(h.x, h.y, h.z)
        dz   = z_idx - z_c
        dphi = phi_idx - phi_c

        if abs(dz) >= patch_half or abs(dphi) >= patch_half:
            continue

        iz   = dz   + patch_half
        iphi = dphi + patch_half

        # apply optional EDep threshold
        edep_val = h.edep if h.edep >= EDEP_THRESHOLD else 0.0

        layer = get_layer_AB(h)
        patch[layer, iz, iphi] += edep_val

        hits_inside_patch.append(h)

        # signal PID check
        if h.pid in signal_pids:
            patch_has_signal = True

    if len(hits_inside_patch) == 0:
        return None, None

    # ======================================================
    # DROP PATCH if *all* hits have PIDs in noise_pids
    # ======================================================
    all_noise = True
    for h in hits_inside_patch:
        if (h.pid not in noise_pids) and (h.pid is not None):
            all_noise = False
            break

    if all_noise:
        return None, None

    return patch, patch_has_signal


# ============================================================
# Process ROOT file
# ============================================================
def process_file(filename, patch_size, max_events):
    print(f"Processing: {filename}")
    reader = root_io.Reader(filename)
    events = reader.get("events")

    patches = []
    labels  = []

    event_counter = 0

    for event in events:
        if max_events is not None and event_counter >= max_events:
            break

        event_hits = []
        particles  = defaultdict(list)

        # ----------------------------------------------------
        # Load ALL hits on layer 1 (A+B)
        # ----------------------------------------------------
        for hit in event.get("VertexBarrelCollection"):
            try:
                if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                    continue

                pos = hit.getPosition()
                mc  = hit.getMCParticle()

                # keep hit even if mc is None
                pid     = mc.getPDG() if mc is not None else None
                trackID = mc.getObjectID().index if mc is not None else -1
                energy  = mc.getEnergy() if mc is not None else 0.0

                try:
                    edep = hit.getEDep()
                except AttributeError:
                    edep = 0.0

                h = functions.Hit(
                    x=pos.x, y=pos.y, z=pos.z,
                    energy=energy, edep=edep,
                    trackID=trackID
                )
                h.pid = pid

                event_hits.append(h)
                particles[trackID].append(h)

            except Exception as e:
                print("Hit error:", e)
                continue

        # ----------------------------------------------------
        # Build patch for each particle cluster
        # ----------------------------------------------------
        for trackID, hitlist in particles.items():
            if len(hitlist) == 0:
                continue

            bx, by, bz = functions.geometric_baricenter(hitlist)

            patch, has_signal = make_patch_for_particle(
                event_hits, bx, by, bz, patch_size
            )
            if patch is None:
                continue

            label = 1 if has_signal else 0
            patches.append(patch)
            labels.append(label)

        event_counter += 1

    return patches, labels


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="AB_patches_final.npz")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    patch_size = args.patch_size
    if patch_size % 2 != 0:
        raise ValueError("Patch size must be even (for centering).")

    all_patches = []
    all_labels  = []

    for name, cfg in SAMPLES.items():
        print(f"\n===== SAMPLE: {name.upper()} =====")
        files = cfg["files"]
        max_events = EVENT_LIMITS.get(name, None)

        if args.max_files is not None:
            files = files[:args.max_files]

        for i, f in enumerate(files):
            print(f"File {i+1}/{len(files)}")
            patches, labels = process_file(f, patch_size, max_events)
            all_patches.extend(patches)
            all_labels.extend(labels)

    X = np.stack(all_patches, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, X=X, y=y)

    print("\n==== DONE ====")
    print("Saved:", args.out)
    print("Total patches:", len(X))
    print("Patch shape:", X.shape[1:])
    print("Class balance:", np.unique(y, return_counts=True))


if __name__ == "__main__":
    main()
