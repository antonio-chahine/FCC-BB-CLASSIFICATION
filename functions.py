import boost_histogram as bh
from matplotlib.ticker import ScalarFormatter
from podio import root_io
import ROOT
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import matplotlib.pyplot as plt
import mplhep as hep 
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import leastsq
import random

hep.style.use(hep.style.ROOT)

#This is the sensor size. Modules are squares of 256 sensors 0.025mm x 0.025mm (25µm) 
PITCH_MM = 0.025
#Layer of desired radius       
RADIUS_MM = 14  
#The length of the z range desired
max_z = 110 # CLD first layer 
#Computes the number of φ-rows (bins) around cylindrical detector layer given sensor pitch.
n_phi_bins = int((2 * math.pi * RADIUS_MM) / PITCH_MM)  
#Computes the number of z-rows (bins) spanning the detector layer given sensor pitch.
n_z_bins = int((2 * max_z) / PITCH_MM)

OVERLAP_CENTERS = [(13.045, 3.723), (10.631, 8.433), (6.594, 11.859), (1.554, 13.478), (-3.724, 13.046), (-8.433, 10.630), (-11.859, 6.594), (-13.481, 1.555),
    (-13.044, -3.724), (-10.629, -8.433), (-6.594, -11.858), (-1.555, -13.478), (3.722, -13.050), (8.434, -10.629), (11.857, -6.595), (13.480, -1.554),]

OVERLAP_RADIUS = 0.2  # mm – to match cluster barycenter to a zone
MERGE_RADIUS = 0.6    # mm – to merge nearby hits

@dataclass
class Hit:
    x: float
    y: float
    z: float
    energy: float
    edep: float
    #charge: float
    trackID: int
    def r(self):
        return math.sqrt(self.x**2 + self.y**2)

    def phi(self):
        return math.atan2(self.y, self.x) % (2 * math.pi)
    
@dataclass
class Particle:
    trackID: int
    """cellID: int
    pid: int"""
    hits: List[Hit] = field(default_factory=list)

    def add_hit(self, hit: Hit):
        self.hits.append(hit)

    def total_energy(self):
        return sum(h.edep for h in self.hits)
    
    #def total_charge(self):
    #    return sum(h.charge for h in self.hits)

    def phi_spread(self) -> float:
        """For 2-hit clusters, return the minimal arc span ∈ [0, π], i.e. correct for wrap-around.
        For clusters with ≥3 hits, return the naïve span (max-min),
        which can legitimately exceed π if the cluster truly covers more than half the circle."""
        phis: List[float] = [h.phi() for h in self.hits]
        if len(phis) < 2:
            return 0.0
        span = max(phis) - min(phis)
        if len(phis) == 2 and span > math.pi:
            # two points: use the shorter wrap-around arc
            span = 2*math.pi - span
        return span

    def z_extent(self):
        if not self.hits:
            return 0
        zs = [h.z for h in self.hits]
        return max(zs) - min(zs)
    
    def r_extent(self):
        if not self.hits:
            return 0
        rs = [h.r() for h in self.hits]
        return max(rs) - min(rs)

    def n_phi_rows(self, pitch_mm: float, radius_mm: float):
        rows = set()
        for h in self.hits:
            arc = radius_mm * h.phi()
            row = int(arc / pitch_mm)
            rows.add(row)
        return len(rows)
    
def calc_phi(x, y):
    """
    Computes azimuthal angle φ in radians ∈ [0, 2π),
    measured in the x-y plane around the beam axis.
    """
    return math.atan2(y, x) % (2 * math.pi)
     
def theta(x,y,z):
    """
    Calculates theta of particle.
    Inputs: x,y,z floats.
    Output: theta, float representing angle in radians from 0 to pi.
    """
    return math.acos(z/np.sqrt(x**2 + y**2 + z**2))


def radius_idx(hit, layer_radii):
    """
    Calculates polar radius of particle.
    Inputs: hit, SimTrackerHit object.
    Output: r, int representing polar radius in mm.
    """
    true_radius = hit.rho()
    for i,r in enumerate(layer_radii):
        if abs(true_radius-r) < 4:
            return i
    raise ValueError(f"Not close enough to any of the layers {true_radius}")

def get_grid_indices(x, y, z):
    """
    Returns (z_index, phi_index) for 25 μm binning in (z, φ arc) space.
    z ∈ [-z_range, +z_range] maps to bins [0, 2*z_range / pitch]
    """
    phi_val = calc_phi(x,y)
    arc_length = RADIUS_MM * phi_val
    phi_index = int(arc_length / PITCH_MM)
    z_index = int((z + max_z) / PITCH_MM)
    
    return z_index, phi_index

    
def compute_centroid(hits: List["Hit"]) -> Tuple[float, float, float]:
    """
    Compute the (x, y, z) centroid of a list of hits.
    """
    if not hits:
        return (0.0, 0.0, 0.0)
    x = sum(h.x for h in hits) / len(hits)
    y = sum(h.y for h in hits) / len(hits)
    z = sum(h.z for h in hits) / len(hits)
    return (x, y, z)

def compute_elongation_phi_z(hits: List["Hit"], radius_mm: float) -> float:
    """
    Perform PCA on arc-length vs z and return elongation (λ₁ / λ₂), guaranteed ≥ 1.
    """
    if len(hits) < 2:
        return None  # no shape with fewer than 2 hits

    # Compute φ arc lengths and z coordinates
    arc_lengths = [radius_mm * h.phi() for h in hits]
    zs = [h.z for h in hits]
    coords = np.vstack((arc_lengths, zs))  # shape: (2, N)

    # Compute covariance matrix and eigenvalues
    cov = np.cov(coords)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]  # λ₁ ≥ λ₂

    if eigvals[1] <= 0:
        return float('inf')  # avoid division by zero
    return eigvals[0] / eigvals[1]

def compute_bounding_box_phi_z(hits: List["Hit"], radius_mm: float) -> Tuple[float, float, float, float]:
    """
    Return (min_arc, max_arc, min_z, max_z) for cluster extent in φ-z space.
    """
    if not hits:
        return (0.0, 0.0, 0.0, 0.0)

    arcs = [radius_mm * h.phi() for h in hits]
    zs = [h.z for h in hits]

    arc_min = min(arcs)
    arc_max = max(arcs)
    z_min = min(zs)
    z_max = max(zs)

    return (arc_min, arc_max, z_min, z_max)

def compute_dphi_dz_two_hits(particles):
    dphis, dzs = [], []
    for p in particles:
        if len(p.hits) == 2:
            h1, h2 = p.hits
            dphi = abs(h1.phi() - h2.phi())
            if dphi > math.pi:
                dphi = 2 * math.pi - dphi
            dz = abs(h1.z - h2.z)
            dphis.append(dphi)
            dzs.append(dz)
    return dphis, dzs

def geometric_baricenter(hits):
    #Iterates over an array of hits and calculates their barycenter.
    sum_energy = 0.0
    sum_energy_x = 0.0
    sum_energy_y = 0.0
    sum_energy_z = 0.0
    for h in hits:
        edep = h.edep
        x = h.x
        y = h.y
        z = h.z
        sum_energy += edep
        sum_energy_x += edep*x
        sum_energy_y += edep*y
        sum_energy_z += edep*z
    b_x = sum_energy_x/sum_energy
    b_y = sum_energy_y/sum_energy
    b_z = sum_energy_z/sum_energy
    return [b_x, b_y, b_z]

def cos_theta(x,y,z):
    norm = math.sqrt(x**2 + y**2 + z**2)
    return z / norm if norm != 0 else None  # Avoid division by zero

def discard_AB(pos):
    r = math.sqrt((pos.x)**2 + (pos.y)**2) 
    if r > 13.0 and r < 14.0:
        return False
    else:
        return True
    
    
####################Do overlap layer merging by summing energy and keeping innermost radius hit#####################
def merge_cluster_hits(hits: List['Hit']) -> List['Hit']:
    """
    Merge hits in a cluster if the cluster is within a known overlap zone.
    Keeps the (x, y, z) coordinates of the hit with smallest radius in each merge group.

    Parameters:
        hits (List[Hit]): A list of Hit objects (from one Particle)

    Returns:
        List[Hit]: Possibly merged list of Hit objects
    """
    if len(hits) < 2:
        return hits  # No merging needed

    # Compute cluster barycenter
    bx = np.mean([h.x for h in hits])
    by = np.mean([h.y for h in hits])

    # Check if this cluster is in a known overlap zone
    for cx, cy in OVERLAP_CENTERS:
        if math.hypot(bx - cx, by - cy) < OVERLAP_RADIUS:
            return _merge_hits(hits)

    return hits  # Leave untouched if not in overlap zone

def _merge_hits(hits: List['Hit']) -> List['Hit']:
    """Merge hits that are spatially close together in x-y.
    Keeps the coordinates of the hit with smallest radius, and averages edep."""
    merged = []
    used = set()

    for i, h1 in enumerate(hits):
        if i in used:
            continue

        best_hit = h1
        edep_total = h1.edep
        count = 1  # count of merged hits including h1

        for j, h2 in enumerate(hits[i+1:], start=i+1):
            if j in used:
                continue
            if math.hypot(h2.x - h1.x, h2.y - h1.y) < MERGE_RADIUS:
                edep_total += h2.edep
                count += 1
                used.add(j)
                if h2.r() < best_hit.r():
                    best_hit = h2

        edep_avg = edep_total / count
        merged_hit = Hit(x=best_hit.x, y=best_hit.y, z=best_hit.z,
                         energy=best_hit.energy, edep=edep_avg, trackID=best_hit.trackID)
        merged.append(merged_hit)
        used.add(i)

    return merged

def compute_radius_of_curvature(xs, ys):
    """
    Estimate radius of curvature R for a set of (x, y) cluster hit coordinates.
    
    Parameters:
    - xs: array-like, x coordinates of hits
    - ys: array-like, y coordinates of hits

    Returns:
    - R: estimated radius of curvature
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    def calc_R(center):
        xc, yc = center
        return np.sqrt((xs - xc)**2 + (ys - yc)**2)

    def objective(center):
        Ri = calc_R(center)
        return Ri - Ri.mean()

    center_estimate = (xs.mean(), ys.mean())
    center_opt, _ = leastsq(objective, center_estimate)
    Ri = calc_R(center_opt)
    R = Ri.mean()
    return R



################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
######################## Plotting Below#########################################################################################################
def plot_hist(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, xLabel="", yLabel="Events", logY=False):
    fig = plt.figure()
    ax = fig.subplots()

    hep.histplot(h, label="", ax=ax, yerr=False)

    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    #ax.legend(fontsize='x-small')
    if logY:
        ax.set_yscale("log")

    if xMin != -1 and xMax != -1:
        ax.set_xlim([xMin, xMax])
    if yMin != -1 and yMax != -1:
        ax.set_ylim([yMin, yMax])


    fig.savefig(outname, bbox_inches="tight")
    #fig.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
    fig.savefig(outname.replace(".png"), bbox_inches="tight")

def plot_2dhist(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, xLabel="", yLabel="Events", logY=False):
    fig = plt.figure()
    ax = fig.subplots()
    hep.hist2dplot(h, label="", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    #ax.legend(fontsize='x-small')
    if logY:
        ax.set_yscale("log")
    if xMin != -1 and xMax != -1:
        ax.set_xlim([xMin, xMax])
    if yMin != -1 and yMax != -1:
        ax.set_ylim([yMin, yMax])
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)
    
def plot_hist_clusters(data, name, title, xlabel, bins=100, logy=False, logx=False, outdir="."):
    """
    Plot histogram of cluster metrics. Supports log-scale on y- and x-axis.
    """
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots()

    ax.hist(data, bins=bins, histtype='stepfilled', alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Clusters")
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"))
   #fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    
def plot_energy_vs_metric(energies, values, name, title, xlabel, ylabel, logx=False, logy=False, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(energies, values, alpha=0.5, s=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    #fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    
def plot_overlay(x_sig, y_sig, x_bkg, y_bkg, name, xlabel, ylabel,
                 logx=False, logy=False, outdir=".",
                 label_sig="Signal", label_bkg="Background",
                 color_sig="blue", color_bkg="red"):
    fig, ax = plt.subplots()

    if len(x_bkg) > 0 and len(y_bkg) > 0:
        ax.scatter(x_bkg, y_bkg, s=2, alpha=0.5, label=label_bkg, color=color_bkg)
    if len(x_sig) > 0 and len(y_sig) > 0:
        ax.scatter(x_sig, y_sig, s=2, alpha=0.5, label=label_sig, color=color_sig)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(name.replace("_", " ").title())

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close(fig)
        
def plot_hist_difference(data_sig, data_bkg, name, title, xlabel, bins, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    hist_sig, bin_edges = np.histogram(data_sig, bins=bins)
    hist_bkg, _         = np.histogram(data_bkg, bins=bins)
    hist_diff = hist_sig - hist_bkg
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    fig, ax = plt.subplots()
    ax.bar(bin_centers, hist_diff, width=np.diff(bin_edges), align='center', alpha=0.7, label='Signal - Background')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Δ Counts")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_diff.png"))
    #fig.savefig(os.path.join(outdir, f"{name}_diff.pdf"))
    plt.close(fig)
    
def plot_sig_bkg_hist(sig_data, bkg_data, name, title, xlabel, bins, logx=False, logy=False, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots()

    ax.hist(bkg_data, bins=bins, histtype='step', linewidth=1.5, label='Background', color='tab:blue')
    ax.hist(sig_data, bins=bins, histtype='step', linewidth=1.5, label='Signal', color='tab:orange')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_overlay.png"))
    #fig.savefig(os.path.join(outdir, f"{name}_overlay.pdf"))
    plt.close(fig)
    
def extract(clusters, *indices):
    return zip(*[
        tuple(c[i] for i in indices)
        for c in clusters
        if all(c[i] is not None for i in indices)
    ]) if clusters else ([], [])
    
def plot_sig_bkg_3dscatter(sig_x, sig_y, sig_z, bkg_x, bkg_y, bkg_z,
                            name, title, xlabel, ylabel, zlabel,
                            logx=False, logy=False, logz=False, outdir="."):
    os.makedirs(outdir, exist_ok=True)

    # Handle log transforms manually (since 3D axes don't support set_*scale)
    def log_safe(arr):
        return np.log10(np.clip(arr, 1e-12, None))  # Avoid log(0)
    
    sig_x_plot = log_safe(sig_x) if logx else sig_x
    sig_y_plot = log_safe(sig_y) if logy else sig_y
    sig_z_plot = log_safe(sig_z) if logz else sig_z

    bkg_x_plot = log_safe(bkg_x) if logx else bkg_x
    bkg_y_plot = log_safe(bkg_y) if logy else bkg_y
    bkg_z_plot = log_safe(bkg_z) if logz else bkg_z

    # 🔁 Rotate 180° about z-axis
    #sig_x_plot = [-x for x in sig_x_plot]
    sig_y_plot = [-y for y in sig_y_plot]
    #bkg_x_plot = [-x for x in bkg_x_plot]
    bkg_y_plot = [-y for y in bkg_y_plot]

    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(sig_x_plot, sig_y_plot, sig_z_plot, c='blue', label='Signal', alpha=0.5, s=10)
    ax.scatter(bkg_x_plot, bkg_y_plot, bkg_z_plot, c='red', label='Background', alpha=0.5, s=10)

    # Axis labels with log indication
    def log_label(label, is_log):
        return f"log10({label})" if is_log else label

    ax.set_xlabel(log_label(xlabel, logx))
    ax.set_ylabel(log_label(ylabel, logy))
    ax.set_zlabel(log_label(zlabel, logz))
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}.png", dpi=300)
    plt.close()

def plot_energy_vs_costheta_binned(sig_costheta, sig_edep, bkg_costheta, bkg_edep, nbins=10, bins=None, outdir="plots_cos_theta_energy"):
    sig_abs = np.abs(sig_costheta)
    bkg_abs = np.abs(bkg_costheta)

    if bins is None:
        bins = np.linspace(0, 1, nbins + 1)

    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_label = f"|cosθ| [{bin_min:.2f}, {bin_max:.2f})"

        # Apply bin mask
        sig_mask = (sig_abs >= bin_min) & (sig_abs < bin_max)
        bkg_mask = (bkg_abs >= bin_min) & (bkg_abs < bin_max)

        sig_in_bin = np.array(sig_edep)[sig_mask]
        bkg_in_bin = np.array(bkg_edep)[bkg_mask]

        print(f"Bin {i}: [{bin_min}, {bin_max}) → Signal: {len(sig_in_bin)}, Background: {len(bkg_in_bin)}")

        # Skip empty bins
        if len(sig_in_bin) == 0 and len(bkg_in_bin) == 0:
            continue

        name = f"edep_vs_abscostheta_bin_{i}"
        title = f"Energy Deposit for {bin_label}"
        xlabel = "Energy Deposit (GeV)"

        plot_sig_bkg_hist(sig_in_bin, bkg_in_bin, name=name, title=title, xlabel=xlabel, bins=np.logspace(-6, -1, 50), logx=True, logy=True, outdir=outdir)
               
def plot_dz_vs_costheta_per_multiplicity(sig_costheta, sig_dz, sig_mult, bkg_costheta, bkg_dz, bkg_mult, multiplicities=None, outdir="vtx_comparison_edep_plots/plots_dz_vs_costheta_by_mult"):

    os.makedirs(outdir, exist_ok=True)

    sig_costheta = np.array(sig_costheta)
    sig_dz = np.array(sig_dz)
    sig_mult = np.array(sig_mult)
    bkg_costheta = np.array(bkg_costheta)
    bkg_dz = np.array(bkg_dz)
    bkg_mult = np.array(bkg_mult)

    if multiplicities is None:
        all_mults = np.unique(np.concatenate([sig_mult, bkg_mult]))
        multiplicities = sorted(all_mults)

    xlim = [-1.0, 1.0]
    ylim = [1e-5, 3e2]

    for m in multiplicities:
        sig_mask = (sig_mult == m)
        bkg_mask = (bkg_mult == m)

        sig_cos = sig_costheta[sig_mask]
        sig_dz_vals = sig_dz[sig_mask]
        bkg_cos = bkg_costheta[bkg_mask]
        bkg_dz_vals = bkg_dz[bkg_mask]

        print(f"Multiplicity {m}: Signal: {len(sig_cos)}, Background: {len(bkg_cos)}")
        if len(sig_cos) == 0 and len(bkg_cos) == 0:
            continue

        fig, ax = plt.subplots()
        ax.scatter(bkg_cos, bkg_dz_vals, s=2, alpha=0.5, label="Background", color='red')
        ax.scatter(sig_cos, sig_dz_vals, s=2, alpha=0.5, label="Signal", color='blue')
        ax.set_xlabel("cos(θ)")
        ax.set_ylabel(r"$\Delta z$ [mm]")
        ax.set_title(f"dz_vs_costheta_mult_{m}".replace("_", " ").title())
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"dz_vs_costheta_mult_{m}.png"))
        plt.close(fig)    

def plot_cluster_scatter(phi, z, outname, cluster_type, cluster_id, cos_theta, multiplicity):
    """
    Simple φ-z scatter plot with φ ∈ [−π, π], no scientific axis confusion.
    """
    fig, ax = plt.subplots()

    # Wrap φ ∈ [−π, π]
    phi = np.array(phi)
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    z = np.array(z)

    # Plot
    color = 'blue' if cluster_type == "signal" else 'red'
    ax.scatter(z, phi, s=100, alpha=0.5, color=color)

    # Labels
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("φ [rad]")
    ax.set_title(f"{cluster_type.title()} Cluster {cluster_id}")

    # Fix axis formatting
    ax.ticklabel_format(style='plain', axis='y')  # Disable sci-notation for φ
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(outname)
    plt.close(fig)
     
def plot_hitmap(hits, outname="hitmap_with_barycenter.png", bins=100):
    # Convert hits to arrays
    xs = [h.x for h in hits]
    ys = [h.y for h in hits]
    edeps = [h.edep for h in hits]

    # Define histogram range
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Create 2D ROOT histogram
    hist2d = ROOT.TH2D("hitmap", "", bins, x_min, x_max, bins, y_min, y_max)
    for x, y, e in zip(xs, ys, edeps):
        hist2d.Fill(x, y, e)

    # Compute barycenter
    b_x, b_y, _ = geometric_baricenter(hits)

    # Create canvas
    canvas = ROOT.TCanvas("canvas", "", 800, 600)
    hist2d.GetXaxis().SetTitle("x [mm]")
    hist2d.GetYaxis().SetTitle("y [mm]")
    hist2d.SetTitle("Hitmap with Barycenter")
    hist2d.SetStats(0)
    hist2d.Draw("COLZ")

    # Draw red marker at barycenter
    marker = ROOT.TMarker(b_x, b_y, 29)  # star shape
    marker.SetMarkerColor(ROOT.kRed)
    marker.SetMarkerSize(2.0)
    marker.Draw("SAME")

    # Save plot
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    canvas.SaveAs(outname)
    canvas.Close()
    print(f"Saved: {outname}")
    
    
############################################
##############################################
#########################################
###########################################
############################################
def relabel_noise_clusters(data, noise_pids, energy_cut):
    """Splits data into clean and reassigned-to-background based on PID and MC energy."""
    clean = []
    reassigned = []
    for row in data:
        pid = int(row[8])
        energy = row[4]
        if pid in noise_pids and energy < energy_cut:
            reassigned.append(row)
        else:
            clean.append(row)
    return clean, reassigned

def cellid_to_group(cellID):
    mapping = {
        1: 1,     129: 1,
        8193: 2,  8321: 2,
        16385: 3, 16513: 3,
        24577: 4, 24705: 4,
        32769: 5, 32897: 5,
        40961: 6, 41089: 6,
        49153: 7, 49281: 7,
        57345: 8, 57473: 8,
        65537: 9, 65665: 9,
        73729: 10, 73857: 10,
        81921: 11, 82049: 11,
        90113: 12, 90241: 12,
        98305: 13, 98433: 13,
        106497: 14, 106625: 14,
        114689: 15, 114817: 15,
        122881: 16, 123009: 16
    }
    return mapping.get(cellID)

def get_features_and_labels(signal_data, background_data, epsilon=1e-6, max_samples=None):
    def transform(row):
        z, rows, mult, edep, _, cos, _, _, _ = row
        return (
            math.log(z + epsilon),        # log(z_extent)
            rows,                         # number of φ rows
            mult,                         # multiplicity
            math.log(edep + epsilon),     # log(energy deposition)
            cos                           # cos(θ)
        )
    
    sig = [(1, transform(row)) for row in signal_data]
    bkg = [(0, transform(row)) for row in background_data]
    data = sig + bkg
    random.shuffle(data)

    if max_samples is not None:
        data = data[:max_samples]

    labels, features = zip(*data)
    return np.array(features), np.array(labels)

def plot_feature_importance(importances, feature_names, outdir=".", filename="feature_importance", sort=True):
    # Sanity check
    assert len(importances) == len(feature_names), "Mismatch between importances and names"

    # Optionally sort by importance
    if sort:
        indices = np.argsort(importances)[::-1]
        importances = np.array(importances)[indices]
        feature_names = [feature_names[i] for i in indices]

    # Create square figure
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.bar(range(len(importances)), importances, color="cornflowerblue", edgecolor='black')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Importance", fontsize=12)
    ax.set_title("XGBoost Feature Importances", fontsize=13)

    ax.tick_params(axis='both', which='major', labelsize=11)
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"{filename}.png"), dpi=300)
    plt.close()