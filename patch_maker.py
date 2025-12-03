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

# Geometry from functions.py
PITCH_MM   = functions.PITCH_MM
RADIUS_MM  = functions.RADIUS_MM
MAX_Z      = functions.max_z
N_PHI_BINS = functions.n_phi_bins
N_Z_BINS   = functions.n_z_bins

# Detector config
LAYER_RADII   = [14, 36, 58]
TARGET_LAYER  = 0          # Only hits on first barrel layer (1A+1B)
R_BOUNDARY_AB = 14.0       # A/B split

# Defaults
DEFAULT_PATCH_SIZE = 32
EVENT_LIMITS = {
    "muons":      978,
    "signal":     100,
    "background": 1247,
}

# Input samples
SAMPLES = {
    "muons": {
        "files": glob.glob("/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root"),
        "label": 1,
    },
    "signal": {
        "files": glob.glob("/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root"),
        "label": 1,
    },
    "background": {
        "files": glob.glob("/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root"),
        "label": 0,
    },
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def get_layer_AB(hit):
    r = math.sqrt(hit.x**2 + hit.y**2)
    return 0 if r < R_BOUNDARY_AB else 1


def barycenter_indices(bx, by, bz):
    """Convert barycenter to (z_idx, phi_idx)."""
    return functions.get_grid_indices(bx, by, bz)


def is_center_valid(z_idx, phi_idx, patch_half):
    """Check patch stays fully inside grid."""
    if z_idx < patch_half or z_idx >= (N_Z_BINS - patch_half):
        return False
    if phi_idx < patch_half or phi_idx >= (N_PHI_BINS - patch_half):
        return False
    return True


def make_patch_for_particle(event_hits, bx, by, bz, patch_size):
    """
    Build a patch centred at the cluster barycenter,
    BUT including *all hits* in the event.
    """
    patch_half = patch_size // 2

    z_c, phi_c = barycenter_indices(bx, by, bz)
    if not is_center_valid(z_c, phi_c, patch_half):
        return None

    patch = np.zeros((2, patch_size, patch_size), dtype=np.float32)

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

    if patch.sum() == 0:
        return None

    return patch


# ---------------------------------------------------------------------
# Single ROOT file processing
# ---------------------------------------------------------------------
def process_file(filename, label, sample_name, patch_size, max_events):
    print(f"[{sample_name.upper()}] Reading {filename}")
    reader = root_io.Reader(filename)
    events = reader.get("events")

    patches = []
    labels  = []

    event_counter = 0

    for event in events:
        if max_events is not None and event_counter >= max_events:
            break

        event_hits = []
        particles = {}

        # -------------------------------
        # Load *all* hits on layer 1
        # -------------------------------
        for hit in event.get("VertexBarrelCollection"):
            try:
                # Restrict to layer 1 (A+B)
                if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                    continue

                pos = hit.getPosition()
                mc  = hit.getMCParticle()
                if mc is None:
                    continue

                trackID = mc.getObjectID().index

                # Safe EDep
                try:
                    edep = hit.getEDep()
                except AttributeError:
                    edep = 0.0

                h = functions.Hit(
                    x=pos.x, y=pos.y, z=pos.z,
                    energy=mc.getEnergy(),
                    edep=edep,
                    trackID=trackID,
                )

                event_hits.append(h)

                if trackID not in particles:
                    particles[trackID] = functions.Particle(trackID)
                particles[trackID].add_hit(h)

            except Exception as e:
                print("  Error:", e)

        # -------------------------------
        # Build patches (cluster-centred)
        # -------------------------------
        for p in particles.values():
            if not p.hits:
                continue

            bx, by, bz = functions.geometric_baricenter(p.hits)
            patch = make_patch_for_particle(event_hits, bx, by, bz, patch_size)
            if patch is None:
                continue

            patches.append(patch)
            labels.append(label)

        event_counter += 1

    return patches, labels


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="AB_patches_32x32.npz")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    patch_size = args.patch_size
    if patch_size % 2 != 0:
        raise ValueError("Patch size must be even.")

    all_patches = []
    all_labels = []

    # Process samples
    for sname, cfg in SAMPLES.items():
        files = cfg["files"]
        label = cfg["label"]
        max_events = EVENT_LIMITS.get(sname, None)

        if args.max_files is not None:
            files = files[:args.max_files]

        print(f"\n=== {sname.upper()} ===")

        for i, f in enumerate(files):
            print(f" File {i+1}/{len(files)}")
            patches, lbls = process_file(
                f, label, sname, patch_size, max_events
            )
            all_patches.extend(patches)
            all_labels.extend(lbls)

    # Stack and shuffle
    X = np.stack(all_patches, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    # Save output
    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)
    np.savez(args.out, X=X, y=y)

    print("\nDone.")
    print("Saved:", args.out)
    print("Num patches:", X.shape[0])
    print("Patch shape:", X.shape[1:])
    print("Class balance:", np.unique(y, return_counts=True))


if __name__ == "__main__":
    main()
