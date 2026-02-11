import numpy as np
import matplotlib.pyplot as plt

def _max_component_size(binary):
    """
    Return size of the largest 4-connected component in a 2D boolean array.
    (No scipy needed.)
    """
    H, W = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best = 0

    for r in range(H):
        for c in range(W):
            if not binary[r, c] or visited[r, c]:
                continue

            # BFS/DFS
            stack = [(r, c)]
            visited[r, c] = True
            size = 0


            while stack:
                rr, cc = stack.pop()
                size += 1
                if rr > 0 and binary[rr-1, cc] and not visited[rr-1, cc]:
                    visited[rr-1, cc] = True; stack.append((rr-1, cc))
                if rr < H-1 and binary[rr+1, cc] and not visited[rr+1, cc]:
                    visited[rr+1, cc] = True; stack.append((rr+1, cc))
                if cc > 0 and binary[rr, cc-1] and not visited[rr, cc-1]:
                    visited[rr, cc-1] = True; stack.append((rr, cc-1))
                if cc < W-1 and binary[rr, cc+1] and not visited[rr, cc+1]:
                    visited[rr, cc+1] = True; stack.append((rr, cc+1))

            best = max(best, size)

    return best


def mega_browser(
    X, y,
    cls,
    max_show=100,
    cols=10,
    min_pixels=4,            # require at least N active pixels
    pixel_threshold=0.0,      # pixel counts as "active" if S > this
    min_sum=0.0,              # require total sum >= this
    min_max=0.0,              # require max pixel >= this
    min_blob=0,               # require largest connected blob >= this (0 disables)
    figsize=2.4,
    seed=0,
    save=None
):
    """
    Show up to max_show 'good' patches for a given class.
    Filters:
      - min_pixels of (S > pixel_threshold)
      - min_sum of S
      - min_max of S
      - optional min_blob of largest 4-connected component in (S > pixel_threshold)
    """
    rng = np.random.default_rng(seed)
    idxs = np.where(y == cls)[0]

    good = []
    for idx in idxs:
        S = X[idx].sum(axis=0)  # (H,W)
        active = (S > pixel_threshold)

        n_active = int(active.sum())
        if n_active < min_pixels:
            continue
        if float(S.sum()) < min_sum:
            continue
        if float(S.max()) < min_max:
            continue
        if min_blob > 0:
            if _max_component_size(active) < min_blob:
                continue

        good.append(idx)

    print(f"Found {len(good)} good patches for class {cls} "
          f"(min_pixels={min_pixels}, thr={pixel_threshold}, min_sum={min_sum}, min_max={min_max}, min_blob={min_blob})")

    if len(good) == 0:
        return

    good = np.array(good)
    chosen = rng.choice(good, size=min(max_show, len(good)), replace=False)

    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize, rows * figsize))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, chosen):
        S = X[idx].sum(axis=0)
        im = ax.imshow(S, cmap="inferno")  # keep your per-patch scaling
        ax.set_title(f"idx={idx}", fontsize=10)
        ax.axis("off")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=220)
        print("Saved:", save)
    plt.show()


# -------------------------
# Example usage
# -------------------------
npz = np.load("/work/submit/anton100/msci-project/FCC-BB-CLASSIFICATION/classifier_patch_work/muons_energycut_nomultiplcity_patches_32size_dphifix.npz")
X = npz["X"]
y = npz["y"]

# "Good" = at least 20 active pixels, and at least one connected blob of 8 pixels
mega_browser(X, y, cls=0, max_show=500, cols=10,
             min_pixels=5, pixel_threshold=0.0, min_blob=0,
             seed=1, save="class0_good.png")

mega_browser(X, y, cls=1, max_show=500, cols=10,
             min_pixels=4, pixel_threshold=0.0, min_blob=4,
             seed=1, save="class1_good.png")

print("X.shape:", X.shape)     # expect (N,C,H,W) or (N,H,W,C)
print("y.shape:", y.shape)
print("one patch shape:", X[0].shape)
print("unique classes:", np.unique(y))


def get_good_ids(
    X, y, cls,
    min_pixels=4,
    pixel_threshold=0.0,
    min_sum=0.0,
    min_max=0.0,
    min_blob=0,
    use_abs=True,   # IMPORTANT for your data
):
    idxs = np.where(y == cls)[0]

    ids = []
    Esum = []
    Emax = []
    Npix = []
    Blob = []

    for idx in idxs:
        S = X[idx].sum(axis=0)

        if use_abs:
            active = (np.abs(S) > pixel_threshold)
        else:
            active = (S > pixel_threshold)

        n_active = int(active.sum())
        if n_active < min_pixels:
            continue

        ssum = float(S[S > 0].sum())  # positive energy content (more physical than raw sum)
        smax = float(S.max())

        if ssum < min_sum:
            continue
        if smax < min_max:
            continue

        b = _max_component_size(active) if min_blob > 0 else 0
        if min_blob > 0 and b < min_blob:
            continue

        ids.append(idx)
        Esum.append(ssum)
        Emax.append(smax)
        Npix.append(n_active)
        Blob.append(b)

    ids = np.array(ids, dtype=int)
    Esum = np.array(Esum, dtype=float)
    Emax = np.array(Emax, dtype=float)
    Npix = np.array(Npix, dtype=int)
    Blob = np.array(Blob, dtype=int)

    print(f"class {cls}: {len(ids)} good patches")
    return ids, Esum, Emax, Npix, Blob


def pick_quantile_ids(ids, Esum, quantiles=(0.5, 0.9, 0.99)):
    out = {}
    order = np.argsort(Esum)
    ids_s = ids[order]
    Es_s = Esum[order]

    for q in quantiles:
        k = int(round(q * (len(ids_s) - 1)))
        out[q] = (int(ids_s[k]), float(Es_s[k]))
    return out


npz = np.load("/work/submit/anton100/msci-project/FCC-BB-CLASSIFICATION/classifier_patch_work/muons_energycut_nomultiplcity_patches_32size_dphifix.npz")
X = npz["X"]
y = npz["y"]

# background (class 0)
ids0, E0, M0, N0, B0 = get_good_ids(
    X, y, cls=0,
    min_pixels=4, pixel_threshold=0.0,
    min_blob=0,
    use_abs=True
)

# signal (class 1)
ids1, E1, M1, N1, B1 = get_good_ids(
    X, y, cls=1,
    min_pixels=4, pixel_threshold=0.0,
    min_blob=3,
    use_abs=True
)

q0 = pick_quantile_ids(ids0, E0, quantiles=(0.5, 0.9, 0.99))
q1 = pick_quantile_ids(ids1, E1, quantiles=(0.5, 0.9, 0.99))

print("Class 0 candidates (idx, Esum):", q0)
print("Class 1 candidates (idx, Esum):", q1)



sig_idx = int(ids1[np.argmax(E1)])          # brightest signal
bkg_idx = int(ids0[np.argsort(E0)[len(E0)//2]])  # median background

print("Max contrast pair:")
print("  background idx:", bkg_idx, "Esum:", float(E0[np.argsort(E0)[len(E0)//2]]))
print("  signal idx    :", sig_idx, "Esum:", float(E1.max()))


def print_candidate_table(ids, Esum, Npix=None, title=""):
    order = np.argsort(Esum)
    ids_s = ids[order]
    Es_s  = Esum[order]
    if Npix is not None:
        N_s = Npix[order]
    else:
        N_s = None

    print("\n" + "="*80)
    print(title)
    print("="*80)

    # many quantiles
    qs = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995]
    print("\nQuantile candidates:")
    for q in qs:
        k = int(round(q * (len(ids_s) - 1)))
        if N_s is None:
            print(f" q={q:>6}: idx={int(ids_s[k])}   Esum={Es_s[k]:.3e}")
        else:
            print(f" q={q:>6}: idx={int(ids_s[k])}   Esum={Es_s[k]:.3e}   Npix={int(N_s[k])}")

    # top-K and bottom-K
    K = 15
    print(f"\nLowest {K} Esum:")
    for i in range(min(K, len(ids_s))):
        if N_s is None:
            print(f" {i:>2}: idx={int(ids_s[i])}   Esum={Es_s[i]:.3e}")
        else:
            print(f" {i:>2}: idx={int(ids_s[i])}   Esum={Es_s[i]:.3e}   Npix={int(N_s[i])}")

    print(f"\nHighest {K} Esum:")
    for i in range(1, min(K, len(ids_s)) + 1):
        j = -i
        if N_s is None:
            print(f" {i:>2}: idx={int(ids_s[j])}   Esum={Es_s[j]:.3e}")
        else:
            print(f" {i:>2}: idx={int(ids_s[j])}   Esum={Es_s[j]:.3e}   Npix={int(N_s[j])}")


print_candidate_table(ids0, E0, Npix=N0, title="Class 0 (background) candidates")
print_candidate_table(ids1, E1, Npix=N1, title="Class 1 (signal) candidates")


def quantile_pick(ids, Esum, q):
    order = np.argsort(Esum)
    ids_s = ids[order]
    Es_s  = Esum[order]
    k = int(round(q * (len(ids_s) - 1)))
    return int(ids_s[k]), float(Es_s[k])

def suggest_pairs(ids0, E0, ids1, E1):
    # (bg quantile, sig quantile) pairs that look nice on a poster
    pairs = [
        (0.50, 0.90),
        (0.50, 0.95),
        (0.50, 0.99),
        (0.70, 0.95),
        (0.70, 0.99),
        (0.90, 0.99),   # same "tail vs tail" comparison
    ]
    print("\nSuggested comparison pairs (bg vs signal):")
    for qb, qs in pairs:
        b_idx, bE = quantile_pick(ids0, E0, qb)
        s_idx, sE = quantile_pick(ids1, E1, qs)
        print(f"  bg q={qb:>4} -> idx={b_idx:<7} Esum={bE:.3e}   |   sig q={qs:>4} -> idx={s_idx:<7} Esum={sE:.3e}   ratio={sE/(bE+1e-30):.2e}")

    # maximal contrast: median background vs brightest signal
    b_idx, bE = quantile_pick(ids0, E0, 0.50)
    s_idx = int(ids1[np.argmax(E1)])
    sE = float(E1.max())
    print("\nMax contrast:")
    print(f"  bg median idx={b_idx} Esum={bE:.3e}")
    print(f"  sig max    idx={s_idx} Esum={sE:.3e}   ratio={sE/(bE+1e-30):.2e}")


suggest_pairs(ids0, E0, ids1, E1)


import matplotlib.colors as colors

def save_pair_plot(X, bg_idx, sig_idx, outname, gamma=0.35, q_vmax=0.995):
    S0 = X[bg_idx,0] + X[bg_idx,1]
    S1 = X[sig_idx,0] + X[sig_idx,1]
    S0 = np.clip(S0, 0, None)
    S1 = np.clip(S1, 0, None)

    eps = 1e-8
    vals = np.concatenate([S0[S0>0].ravel(), S1[S1>0].ravel()])
    vmax = float(np.quantile(vals, q_vmax)) if vals.size else 1.0

    norm = colors.PowerNorm(gamma=gamma, vmin=eps, vmax=vmax)

    cmap = plt.cm.inferno.copy()
    cmap.set_under("black")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, S, title, idx in zip(
        axes, [S0, S1],
        ["Class 0 (background)", "Class 1 (signal)"],
        [bg_idx, sig_idx]
    ):
        S = np.where(S > 0, S, eps)
        ax.imshow(S.T, cmap=cmap, norm=norm, origin="lower", aspect="equal")
        ax.set_title(f"{title}\nidx={idx}", fontsize=22)
        ax.set_xlabel("z axis", fontsize=18)
        ax.set_ylabel("φ axis", fontsize=18)
        ax.set_xticks([]); ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axes[0].images[0], cbar_ax)
    cbar.set_label("Energy deposition", fontsize=18)

    plt.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.90, wspace=0.25)
    plt.savefig(outname, dpi=200)
    plt.close(fig)


pairs = [(0.50,0.90), (0.50,0.95), (0.50,0.99), (0.70,0.99)]
for qb, qs in pairs:
    b_idx, _ = quantile_pick(ids0, E0, qb)
    s_idx, _ = quantile_pick(ids1, E1, qs)
    save_pair_plot(X, b_idx, s_idx, f"pair_bg{qb}_sig{qs}.png", gamma=0.35)
print("Saved comparison pairs.")
