import numpy as np
import matplotlib.pyplot as plt

data = np.load("AB_patches_32x32.npz")
X = data["X"]
y = data["y"]

# How many of each label to show
num_signal = 4
num_background = 4

sig_idx = np.where(y == 1)[0]
bkg_idx = np.where(y == 0)[0]

chosen_sig = np.random.choice(sig_idx, num_signal, replace=False)
chosen_bkg = np.random.choice(bkg_idx, num_background, replace=False)

indices = np.concatenate([chosen_sig, chosen_bkg])

# Plotting
fig, axes = plt.subplots(len(indices), 2, figsize=(7, 3*len(indices)))

for i, idx in enumerate(indices):
    patch = X[idx]
    label = y[idx]

    axes[i, 0].imshow(patch[0], cmap="inferno", interpolation="nearest")
    axes[i, 0].set_title(f"Layer A — Label {label}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(patch[1], cmap="inferno", interpolation="nearest")
    axes[i, 1].set_title(f"Layer B — Label {label}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("sample_patches_balanced.png", dpi=300)
plt.show()
