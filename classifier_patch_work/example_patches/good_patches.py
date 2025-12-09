import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# -----------------------------
# Select examples
# -----------------------------
class0_idx = 11836   # background
class1_idx = 92      # signal
indices = [class0_idx, class1_idx]
titles = ["Class 0 (background)", "Class 1 (signal)"]

# -----------------------------
# Load dataset
# -----------------------------
npz = np.load("AB_patches_final.npz")
X = npz["X"]
y = npz["y"]

# -----------------------------
# Compute global vmax (shared colour scale)
# -----------------------------
global_vmax = max(
    (X[class0_idx,0] + X[class0_idx,1]).max(),
    (X[class1_idx,0] + X[class1_idx,1]).max()
)

# -----------------------------
# LogNorm settings
# -----------------------------
eps = 1e-8  # avoid log(0)

cmap = plt.cm.inferno.copy()
cmap.set_under("black")  # zero → black instead of white

# -----------------------------
# Create figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, idx, title in zip(axes, indices, titles):

    patch = X[idx,0] + X[idx,1]
    patch_safe = np.where(patch > 0, patch, eps)

    # -----------------------------
    # Swap axes → z on x-axis, φ on y-axis
    # -----------------------------
    patch_t = patch_safe.T

    im = ax.imshow(
        patch_t,
        cmap=cmap,
        norm=colors.LogNorm(vmin=eps, vmax=global_vmax),
        origin="lower",
        aspect="equal"
    )

    ax.set_title(f"{title}\nidx = {idx}", fontsize=22)

    # -----------------------------
    # AXIS LABELS BUT NO NUMBERS
    # -----------------------------
    ax.set_xlabel("z axis", fontsize=18)
    ax.set_ylabel("φ axis", fontsize=18)

    ax.set_xticks([])   # hide tick numbers
    ax.set_yticks([])

# -----------------------------
# Colourbar (outside both images)
# -----------------------------
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cbar_ax)
cbar.set_label("Energy deposition", fontsize=18)

# -----------------------------
# Layout adjustments
# -----------------------------
plt.subplots_adjust(
    left=0.08, right=0.88,
    bottom=0.12, top=0.90,
    wspace=0.25
)

plt.savefig("phi_z_comparison_clean.png", dpi=200)
plt.show()
