import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

def draw_bdt_schematic(
    outpath="bdt_schematic.png",
    dpi=300,
    seed=2,
):
    rng = np.random.default_rng(seed)

    # -----------------------------
    # Canvas / coordinate system
    # -----------------------------
    # Treat x as "z-like" axis and y as "phi-like" axis for the drawing.
    x_min, x_max = 0, 10
    y_min, y_max = 0, 7

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    # -----------------------------
    # Light dot-grid background
    # -----------------------------
    gx = np.arange(x_min, x_max + 1e-9, 0.5)
    gy = np.arange(y_min, y_max + 1e-9, 0.5)
    X, Y = np.meshgrid(gx, gy)
    ax.scatter(X.ravel(), Y.ravel(), s=4, alpha=0.15)

    # -----------------------------
    # Example "hits" (red points)
    # -----------------------------
    # Make a loose diagonal cluster
    n_hits = 10
    t = np.linspace(0.1, 0.9, n_hits)
    hits_x = 4.5 + 2.0*(t - 0.5) + rng.normal(0, 0.15, n_hits)
    hits_y = 3.6 + 4.0*(t - 0.5) + rng.normal(0, 0.15, n_hits)
    ax.scatter(hits_x, hits_y, s=40, marker="o", edgecolor="none")

    # -----------------------------
    # Local axes (lambda1, lambda2)
    # -----------------------------
    origin = np.array([6.2, 3.7])

    # Choose an orientation (degrees)
    theta = np.deg2rad(65)  # angle of λ1 w.r.t +x
    v1 = np.array([np.cos(theta), np.sin(theta)])      # λ1 direction
    v2 = np.array([-np.sin(theta), np.cos(theta)])     # λ2 direction (perpendicular)

    L1 = 3.0
    L2 = 2.2
    purple = "#7b2cbf"

    # λ1 line (solid)
    p1a = origin - 0.15 * L1 * v1
    p1b = origin + 0.85 * L1 * v1
    ax.plot([p1a[0], p1b[0]], [p1a[1], p1b[1]], lw=2.5, color=purple)

    # λ2 line (solid)
    p2a = origin - 0.55 * L2 * v2
    p2b = origin + 0.55 * L2 * v2
    ax.plot([p2a[0], p2b[0]], [p2a[1], p2b[1]], lw=2.5, color=purple)

    # Labels λ1, λ2
    ax.text(p1b[0] + 0.15, p1b[1] + 0.05, r"$\lambda_1$", color=purple, fontsize=14)
    ax.text(p2b[0] + 0.10, p2b[1] + 0.05, r"$\lambda_2$", color=purple, fontsize=14)

    # Right-angle marker at origin
    ra = 0.35
    corner = origin + 0.05*v1 + 0.05*v2
    A = corner + ra * v1
    B = corner + ra * v1 + ra * v2
    C = corner + ra * v2
    ax.plot([corner[0], A[0]], [corner[1], A[1]], color=purple, lw=2)
    ax.plot([A[0], B[0]], [A[1], B[1]], color=purple, lw=2)
    ax.plot([B[0], C[0]], [B[1], C[1]], color=purple, lw=2)

    # -----------------------------
    # Δz extent (horizontal dimension arrow)
    # -----------------------------
    z_y = 0.9
    z_x0, z_x1 = 5.2, 8.0

    arrow_kw = dict(arrowstyle="|-|", mutation_scale=15, lw=2, color="black")
    ax.add_patch(FancyArrowPatch((z_x0, z_y), (z_x1, z_y), **arrow_kw))
    ax.text((z_x0 + z_x1)/2, z_y - 0.35, r"$\Delta z$", ha="center", va="top", fontsize=13)

    # -----------------------------
    # Δphi extent (vertical dimension arrow)
    # -----------------------------
    phi_x = 2.3
    phi_y0, phi_y1 = 1.2, 6.3
    ax.add_patch(FancyArrowPatch((phi_x, phi_y0), (phi_x, phi_y1), **arrow_kw))
    ax.text(phi_x - 0.35, (phi_y0 + phi_y1)/2, r"$\Delta \phi$",
            rotation=90, ha="right", va="center", fontsize=13)

    # -----------------------------
    # Pixel size and "8 phi bins" annotations
    # -----------------------------
    # Pixel size box in top-left
    ax.add_patch(Rectangle((0.7, 6.25), 0.55, 0.55, fill=False, lw=2, edgecolor="black"))
    ax.text(1.45, 6.52, r"$25\times 25\,\mu\mathrm{m}$", ha="left", va="center", fontsize=13)

    ax.text(8.1, 6.2, r"$N_{\phi}=8$ bins", color="#c0392b", fontsize=16, ha="left")

    plt.tight_layout(pad=0.2)
    fig.savefig(outpath, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    draw_bdt_schematic(outpath="bdt_schematic.png")
