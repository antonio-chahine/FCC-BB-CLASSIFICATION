#!/usr/bin/env python3

import glob
import math
import os
import time
import numpy as np
from collections import defaultdict

from podio import root_io
import ROOT
ROOT.gROOT.SetBatch(True)

import functions

# ============================================================
# USER SETTINGS (edit these)
# ============================================================
OUTDIR = "/ceph/submit/data/user/a/anton100/msci-project"
OUT_PREFIX = "muons_energycut_nomultiplcity_patches_32size_0.00625pixel"  # will create shards
MAX_FILES = None          # e.g. 5 for debugging, or None for all
FLUSH_EVERY = 500         # number of patches per shard
MERGE_TO_SINGLE_NPZ = True  # set True if you really want one big .npz at the end (costly)

# ============================================================
# Geometry / pixel size
# ============================================================
functions.PITCH_MM = 0.00625

PITCH_MM  = functions.PITCH_MM
RADIUS_MM = functions.RADIUS_MM
MAX_Z     = functions.max_z

N_PHI_BINS = int((2 * math.pi * RADIUS_MM) / PITCH_MM)
N_Z_BINS   = int((2 * MAX_Z) / PITCH_MM)

LAYER_RADII   = [14, 36, 58]
TARGET_LAYER  = 0
R_BOUNDARY_AB = 13.919

# keep physical patch width fixed to 32 * 25um
REF_WIDTH_MM = 32 * 0.025
DEFAULT_PATCH_SIZE = int(round(REF_WIDTH_MM / PITCH_MM))
if DEFAULT_PATCH_SIZE % 2 == 1:
    DEFAULT_PATCH_SIZE += 1
DEFAULT_PATCH_SIZE = max(4, DEFAULT_PATCH_SIZE)

PATCH_SIZE = DEFAULT_PATCH_SIZE

noise_pids = {11, -11, 22}
ENERGY_CUT = 0.01

# ============================================================
# Input samples
# ============================================================
SAMPLES = {
    "muons": {
        "files": glob.glob(
            "/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root"
        ),
        "label": 1,
    },
    "signal": {
        "files": glob.glob(
            "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root"
        ),
        "label": 1,
    },
    "background": {
        "files": glob.glob(
            "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root"
        ),
        "label": 0,
    },
}

EVENT_LIMITS = {"muons": 978, "signal": 100, "background": 1247}

BAD_FILES = {
    "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/240499.root",
    "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/444948.root",
    "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/839982.root",
}

# ============================================================
# Helpers
# ============================================================
def get_layer_AB(hit):
    r = math.hypot(hit.x, hit.y)
    return 0 if r < R_BOUNDARY_AB else 1


def classify_patch(patch_hits, source_label):
    # background stays background
    if source_label == 0:
        return 0

    # signal/muons: veto "pure noise + low energy"
    all_noise_pid = all(h.pid in noise_pids for h in patch_hits)
    all_low_energy = all(h.energy < ENERGY_CUT for h in patch_hits)

    if all_noise_pid and all_low_energy:
        return 0

    return 1


def make_patch(event_hits, bx, by, bz, patch_size, source_label):
    patch_half = patch_size // 2

    z_c, phi_c = functions.get_grid_indices(bx, by, bz)

    # fixed physical jitter: 4 pixels at 25um => 0.1mm
    REF_JITTER_MM = 4 * 0.025
    jitter_pix = int(round(REF_JITTER_MM / PITCH_MM))
    z_c   += np.random.randint(-jitter_pix, jitter_pix + 1)
    phi_c += np.random.randint(-jitter_pix, jitter_pix + 1)

    z_c   = int(np.clip(z_c, patch_half, N_Z_BINS - patch_half - 1))
    phi_c = int(np.clip(phi_c, patch_half, N_PHI_BINS - patch_half - 1))

    patch = np.zeros((2, patch_size, patch_size), dtype=np.float32)
    patch_hits = []

    for h in event_hits:
        z_idx, phi_idx = functions.get_grid_indices(h.x, h.y, h.z)

        phi_idx %= N_PHI_BINS
        z_idx = max(0, min(N_Z_BINS - 1, z_idx))

        dz = z_idx - z_c
        dphi = ((phi_idx - phi_c + N_PHI_BINS // 2) % N_PHI_BINS) - N_PHI_BINS // 2

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


# ============================================================
# Sharded saving (safe incremental)
# ============================================================
def atomic_savez(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = path + ".tmp.npz"   # IMPORTANT: ends with .npz
    np.savez(tmp, **arrays)
    os.replace(tmp, path)



def shard_path(outdir, prefix, shard_idx):
    return os.path.join(outdir, f"{prefix}_shard{shard_idx:05d}.npz")


def write_shard(outdir, prefix, shard_idx, Xb, yb, meta):
    os.makedirs(outdir, exist_ok=True)
    path = shard_path(outdir, prefix, shard_idx)
    atomic_savez(path, X=Xb, y=yb, meta=np.array([meta], dtype=object))
    return path


def list_shards(outdir, prefix):
    pat = os.path.join(outdir, f"{prefix}_shard*.npz")
    return sorted(glob.glob(pat))


def summarize_shards(outdir, prefix):
    shards = list_shards(outdir, prefix)
    total = 0
    counts = defaultdict(int)

    for s in shards:
        with np.load(s, allow_pickle=True) as d:
            y = d["y"]
            total += len(y)
            u, c = np.unique(y, return_counts=True)
            for uu, cc in zip(u.tolist(), c.tolist()):
                counts[int(uu)] += int(cc)

    return shards, total, counts


def merge_shards_to_single(outdir, prefix, out_npz_path):
    shards = list_shards(outdir, prefix)
    if not shards:
        raise RuntimeError("No shards found to merge.")

    Xs, ys = [], []
    for s in shards:
        with np.load(s, allow_pickle=True) as d:
            Xs.append(d["X"])
            ys.append(d["y"])

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    atomic_savez(out_npz_path, X=X.astype(np.float32), y=y.astype(np.int8))


# ============================================================
# Process ROOT file
# ============================================================
def process_file(filename, sample_label, sample_name, patch_size, max_events):
    print(f"[{sample_name.upper()}] {filename}")

    reader = root_io.Reader(filename)
    events = reader.get("events")

    patches = []
    labels = []

    for i_event, event in enumerate(events):

        if max_events and i_event >= max_events:
            break

        event_hits = []
        particles = defaultdict(list)

        for hit in event.get("VertexBarrelCollection"):
            if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                continue

            pos = hit.getPosition()
            mc = hit.getMCParticle()

            if mc is None:
                pid = None
                trackID = -1
                energy = 999.0
            else:
                pid = mc.getPDG()
                trackID = mc.getObjectID().index
                energy = mc.getEnergy()

            try:
                edep = hit.getEDep()
            except Exception:
                edep = 0.0

            h = functions.Hit(
                x=pos.x,
                y=pos.y,
                z=pos.z,
                energy=energy,
                edep=edep,
                trackID=trackID,
            )
            h.pid = pid

            event_hits.append(h)
            particles[trackID].append(h)

        for hitlist in particles.values():
            if len(hitlist) == 2:
                hitlist = functions.merge_cluster_hits(hitlist)

            bx, by, bz = functions.geometric_baricenter(hitlist)

            patch, label = make_patch(
                event_hits, bx, by, bz, patch_size, sample_label
            )

            if patch is not None:
                patches.append(patch)
                labels.append(label)

    return patches, labels


# ============================================================
# Main
# ============================================================
def main():
    outdir = OUTDIR
    prefix = OUT_PREFIX

    print("PITCH_MM:", PITCH_MM)
    print("PATCH_SIZE:", PATCH_SIZE)
    print("Writing shards to:", outdir)
    print("Prefix:", prefix)

    buffer_patches = []
    buffer_labels = []
    shard_idx = 0

    for sname, cfg in SAMPLES.items():
        print(f"\n=== {sname.upper()} ===")

        files = cfg["files"]
        label = cfg["label"]
        max_events = EVENT_LIMITS[sname]

        if MAX_FILES is not None:
            files = files[:MAX_FILES]

        for f in files:
            if f in BAD_FILES:
                print(f"Skipping corrupted file: {f}")
                continue

            p, l = process_file(f, label, sname, PATCH_SIZE, max_events)

            buffer_patches.extend(p)
            buffer_labels.extend(l)

            while len(buffer_labels) >= FLUSH_EVERY:
                # flush exactly FLUSH_EVERY so shards are consistent
                take = FLUSH_EVERY
                Xb = np.stack(buffer_patches[:take]).astype(np.float32)
                yb = np.asarray(buffer_labels[:take], dtype=np.int8)

                meta = {
                    "sample": sname,
                    "pitch_mm": PITCH_MM,
                    "patch_size": PATCH_SIZE,
                    "energy_cut": ENERGY_CUT,
                    "timestamp": time.time(),
                    "n_in_shard": int(len(yb)),
                }

                path = write_shard(outdir, prefix, shard_idx, Xb, yb, meta)
                print(f"  -> wrote {path}  (n={len(yb)})")

                shard_idx += 1
                del buffer_patches[:take]
                del buffer_labels[:take]

    # final flush
    if buffer_labels:
        Xb = np.stack(buffer_patches).astype(np.float32)
        yb = np.asarray(buffer_labels, dtype=np.int8)

        meta = {
            "sample": "final_flush",
            "pitch_mm": PITCH_MM,
            "patch_size": PATCH_SIZE,
            "energy_cut": ENERGY_CUT,
            "timestamp": time.time(),
            "n_in_shard": int(len(yb)),
        }

        path = write_shard(outdir, prefix, shard_idx, Xb, yb, meta)
        print(f"  -> wrote {path}  (n={len(yb)})")
        shard_idx += 1

    shards, total, counts = summarize_shards(outdir, prefix)
    print("\n=== DONE ===")
    print("Num shards:", len(shards))
    print("Total patches:", total)
    print("Class balance:", dict(counts))

    if MERGE_TO_SINGLE_NPZ:
        out_npz = os.path.join(outdir, f"{prefix}.npz")
        print("\nMerging shards to:", out_npz)
        merge_shards_to_single(outdir, prefix, out_npz)
        with np.load(out_npz) as d:
            y = d["y"]
            print("Merged patches:", len(y))
            print("Merged class balance:", np.unique(y, return_counts=True))


if __name__ == "__main__":
    main()
