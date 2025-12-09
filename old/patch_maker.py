import numpy as np
import matplotlib.pyplot as plt


def plot_patches(
    X, y,
    n_per_class=12,
    cols=6,
    mode="all",          # "sum", "A", "B", or "all"
    figsize_scale=2.2,
    share_colorbar=True,
    save=None,
):
    """
    Plot many patches with optional energy scale + A/B display.

    Args:
        X: (N, 2, H, W)
        y: labels
        n_per_class: patches per class to display
        cols: number of columns
        mode: 
            "sum" → A+B only
            "A"   → only layer A
            "B"   → only layer B
            "all" → A, B, A+B displayed side-by-side
        share_colorbar: one global colorbar instead of many
        save: filename for saving
    """

    classes = np.unique(y).astype(int)

    # Select patch indices for each class
    selected = {}
    for cls in classes:
        idxs = np.where(y == cls)[0]
        selected[cls] = np.random.choice(idxs, n_per_class, replace=False)

    # Number of images per patch (1 or 3 depending on mode)
    if mode == "all":
        subcols = 3
    else:
        subcols = 1

    total_cols = cols * subcols
    rows = len(classes) * (n_per_class // cols + int(n_per_class % cols != 0))

    fig, axes = plt.subplots(rows, total_cols, figsize=(total_cols * figsize_scale, rows * figsize_scale))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    # Determine global vmin/vmax for consistent energy scale
    if share_colorbar:
        global_min = X.min()
        global_max = X.max()
    else:
        global_min, global_max = None, None

    r_offset = 0  # row offset for each class

    for cls_idx, cls in enumerate(classes):
        chosen = selected[cls]

        for i, idx in enumerate(chosen):
            r = r_offset + i // cols
            base_c = (i % cols) * subcols  # starting column for this patch group

            patch = X[idx]
            A = patch[0]
            B = patch[1]
            S = A + B

            # Which panels to draw?
            images = []
            labels = []

            if mode == "A":
                images = [A]
                labels = ["Layer A"]
            elif mode == "B":
                images = [B]
                labels = ["Layer B"]
            elif mode == "sum":
                images = [S]
                labels = ["A+B"]
            elif mode == "all":
                images = [A, B, S]
                labels = ["A", "B", "A+B"]

            for k, img in enumerate(images):
                ax = axes[r, base_c + k] if images else None

                im = ax.imshow(img, cmap="inferno",
                               vmin=global_min if share_colorbar else img.min(),
                               vmax=global_max if share_colorbar else img.max())

                ax.set_title(f"{labels[k]}  (class {cls})")
                ax.axis("off")

        r_offset += (n_per_class // cols) + int(n_per_class % cols != 0)

    # Add one shared colorbar
    if share_colorbar:
        fig.subplots_adjust(right=0.92)
        cax = fig.add_axes([0.95, 0.1, 0.015, 0.8])
        fig.colorbar(im, cax=cax, label="Energy deposit (a.u.)")

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=200)
        print("Saved:", save)
    plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    npz = np.load("AB_patches_32x32.npz")
    X = npz["X"]
    y = npz["y"]

    plot_patches(
        X, y,
        n_per_class=12,
        cols=4,
        mode="all",             # A, B, and A+B
        figsize_scale=2.5,
        share_colorbar=True,
        save="patch_inspection.png"
    )
