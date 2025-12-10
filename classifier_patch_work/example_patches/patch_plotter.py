import numpy as np
import matplotlib.pyplot as plt

def mega_browser(
    X, y,
    cls,
    max_show=100,
    cols=10,
    min_pixels=1,         # show only patches with >= this many active pixels
    figsize=2.4,
    save=None
):
    """
    Show up to max_show patches for a given class, numbered and scaled individually.
    """

    idxs = np.where(y == cls)[0]

    # Filter
    good = []
    for idx in idxs:
        S = X[idx].sum(axis=0)
        if np.count_nonzero(S) >= min_pixels:
            good.append(idx)

    print(f"Found {len(good)} non-empty patches for class {cls}")

    if len(good) == 0:
        return

    good = np.array(good)

    # Limit to max_show
    if len(good) > max_show:
        chosen = np.random.choice(good, max_show, replace=False)
    else:
        chosen = good

    rows = (len(chosen) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize, rows * figsize))
    axes = axes.flatten()

    for ax, idx in zip(axes, chosen):
        S = X[idx].sum(axis=0)

        # individual colour scaling improves visibility
        im = ax.imshow(S, cmap='inferno')
        ax.set_title(f"idx={idx}", fontsize=10)
        ax.axis('off')

    # Hide unused
    for ax in axes[len(chosen):]:
        ax.axis('off')

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=220)
        print("Saved:", save)

    plt.show()


# -------------------------
# Example usage
# -------------------------
npz = np.load("/work/submit/anton100/msci-project/FCC-BB-CLASSIFICATION/classifier_patch_work/AB_patches_final_2.npz")
X = npz["X"]
y = npz["y"]

# Show lots of class 0 patches
mega_browser(X, y, cls=0, max_show=100, cols=10, min_pixels=1,
             save="class0_mega.png")

# Show lots of class 1 patches
mega_browser(X, y, cls=1, max_show=100, cols=10, min_pixels=1,
             save="class1_mega.png")
