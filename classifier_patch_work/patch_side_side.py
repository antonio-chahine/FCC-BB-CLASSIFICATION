import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# -----------------------------
# Select examples
# -----------------------------
class0_idx = 219482   # background
class1_idx = 122302   # signal
indices = [class0_idx, class1_idx]
titles = ["Background", "Signal"]

# -----------------------------
# Load dataset
# -----------------------------
npz = np.load("muons_energycut_nomultiplcity_patches_32size_dphifix.npz")
X = npz["X"]
y = npz["y"]

# -----------------------------
# Compute global vmax (shared colour scale)
# -----------------------------
global_vmax = max(
    (X[class0_idx, 0] + X[class0_idx, 1]).max(),
    (X[class1_idx, 0] + X[class1_idx, 1]).max()
)

# -----------------------------
# LogNorm settings
# -----------------------------
eps = 1e-8  # avoid log(0)
cmap = plt.cm.Blues.copy()

# -----------------------------
# Create figure — 1 row, 2 columns
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Example Patches",
    fontsize=20,
    fontweight="bold",
    y=1.01,
    x=0.5
)

for ax, idx, title in zip(axes, indices, titles):
    patch = X[idx, 0] + X[idx, 1]
    patch_masked = np.ma.masked_where(patch == 0, patch)
    patch_t = patch_masked.T

    im = ax.imshow(
        patch_t,
        cmap=cmap,
        norm=colors.LogNorm(vmin=eps, vmax=global_vmax),
        origin="lower",
        aspect="equal"
    )
    ax.set_title(f"{title}", fontsize=18)
    ax.set_xlabel("z axis", fontsize=15)
    ax.set_ylabel("φ axis", fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])

# -----------------------------
# Single shared colourbar to the right of both plots
# -----------------------------
cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
cbar = fig.colorbar(im, cbar_ax)
cbar.ax.tick_params(
    axis='y',
    which='both',
    left=False,
    right=True,
    labelleft=False,
    labelright=True,
    labelsize=14
)
cbar.ax.yaxis.tick_right()
cbar.ax.yaxis.set_label_position('right')
cbar.set_label(
    "Energy deposition (MeV)",
    fontsize=18,
    labelpad=18
)

# -----------------------------
# Layout adjustments
# -----------------------------
plt.subplots_adjust(
    left=0.06, right=0.90,
    bottom=0.10, top=0.92,
    wspace=0.25
)

plt.savefig("phi_z_comparison_sidebyside.pdf", dpi=200, bbox_inches="tight")
plt.show()