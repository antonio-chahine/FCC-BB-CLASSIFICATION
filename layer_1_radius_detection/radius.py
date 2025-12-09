#!/usr/bin/env python3
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from podio import root_io
import ROOT
ROOT.gROOT.SetBatch(True)

from sklearn.cluster import KMeans  # <-- needs scikit-learn

# --- read all radii as before ---
files = []
files += glob.glob("/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root")
files += glob.glob("/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root")
files += glob.glob("/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root")

radii = []

for f in files:
    print("Reading:", f)
    reader = root_io.Reader(f)
    events = reader.get("events")

    for event in events:
        for hit in event.get("VertexBarrelCollection"):
            pos = hit.getPosition()
            r = math.sqrt(pos.x**2 + pos.y**2)
            radii.append(r)

radii = np.array(radii)
print("Total hits:", len(radii))

# -------------------------------------------------
# 1) Focus only on the first layer (1A + 1B)
#    Choose a window that covers only the first clump.
#    Adjust [r_min, r_max] if needed after you look.
# -------------------------------------------------
r_min, r_max = 10.0, 20.0   # rough window for layer 1
mask_L1 = (radii > r_min) & (radii < r_max)
r_L1 = radii[mask_L1]

print("Layer-1 hits:", len(r_L1))

plt.figure(figsize=(6,4))
plt.hist(r_L1, bins=200, histtype="stepfilled", alpha=0.7)
plt.xlabel("Hit radius r [mm]")
plt.ylabel("Counts")
plt.title("Layer-1 hit radii (1A + 1B)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hit_radii_layer1.png", dpi=300)

# -------------------------------------------------
# 2) Find two clusters in radius (1A vs 1B)
#    and define boundary as midpoint of centres
# -------------------------------------------------
r_L1_reshaped = r_L1.reshape(-1, 1)

kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(r_L1_reshaped)

centres = np.sort(kmeans.cluster_centers_.flatten())
rA, rB = centres   # rA < rB
r_boundary = 0.5 * (rA + rB)

print(f"Estimated 1A centre: {rA:.3f} mm")
print(f"Estimated 1B centre: {rB:.3f} mm")
print(f"Suggested 1A/1B boundary: r = {r_boundary:.3f} mm")

# optional: overplot boundary
plt.axvline(r_boundary, linestyle="--")
plt.savefig("hit_radii_layer1_with_boundary.png", dpi=300)
plt.close()
