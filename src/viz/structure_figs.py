"""Figures of the 3D structure diagnostics (each has at most 3 panels).

- Force-field checks (each figure <= 3 panels): bonds and angles, excluded volume, loop springs.
"""
import numpy as np

from analysis.structure import BOND_K, BOND_LENGTH
from reporting.explanations import EXPLAIN
from viz.theme import A_RED, AQUA, B_BLUE, glow, make_fig, MUTED, ORANGE, save_figure, VIOLET


# ---- Figures of the 3D structure diagnostics (each has at most 3 panels) ----

_TXT = lambda k: dict(title=EXPLAIN[k]["title"], caption=EXPLAIN[k]["caption"], explain=EXPLAIN[k]["explain"], section="structure_3d")


def _t(md, n):
    """MD time axis (ns) for n frames."""
    return (np.arange(n) + 1) * md.meta["ps_per_frame"] / 1000.0


def shape_figure(m, md):
    """Size, shape anisotropy and convex-hull packing over the trajectory."""
    s, t = m.shape, _t(md, len(m.shape["rg"]))
    fig, axs = make_fig(1, 3, cell=(4.8, 4.0))
    glow(axs[0], t, s["rg"], VIOLET, label="R_g")
    glow(axs[0], t, s["end_to_end"] / 4, ORANGE, lw=1.2, label="end-to-end / 4")
    axs[0].set(title="Size", xlabel="MD time (ns)", ylabel="Length (nm)")
    axs[0].legend(fontsize=8)
    glow(axs[1], t, s["anisotropy"], AQUA, label="shape anisotropy")
    glow(axs[1], t, s["sphericity"], VIOLET, lw=1.2, label="hull sphericity")
    axs[1].set(title="Shape", xlabel="MD time (ns)", ylabel="Dimensionless (0-1)", ylim=(0, 1))
    axs[1].legend(fontsize=8)
    glow(axs[2], t, s["hull_volume"], ORANGE)
    axs[2].set(title="Convex hull volume", xlabel="MD time (ns)", ylabel="Volume (nm$^3$)")
    save_figure(fig, "struct_shape", **_TXT("struct_shape"))


def pairs_figure(m, md):
    """Pair-distance distribution, nearest-neighbour packing and radial density."""
    fig, axs = make_fig(1, 3, cell=(4.8, 4.0))
    mid = (m.dist_bins[1:] + m.dist_bins[:-1]) / 2
    axs[0].fill_between(mid, m.dist_pdf, color=VIOLET, alpha=0.25)
    glow(axs[0], mid, m.dist_pdf, VIOLET, lw=1.2)
    axs[0].set(title="Pair-to-pair distances", xlabel="Distance between two beads (nm)", ylabel="Probability density (1/nm)")
    t = _t(md, len(m.shape["rg"]))[m.pair_frames]
    glow(axs[1], t, m.pairs["nearest"], AQUA, label="nearest non-bonded neighbour")
    glow(axs[1], t, m.pairs["closest"], ORANGE, lw=1.2, label="closest non-bonded pair")
    glow(axs[1], _t(md, len(m.bonds["bond_mean"])), m.bonds["bond_mean"], VIOLET, lw=1.2, label="bond length")
    axs[1].set(title="Packing and chain integrity", xlabel="MD time (ns)", ylabel="Distance (nm)")
    axs[1].legend(fontsize=8)
    r = (m.r_edges[1:] + m.r_edges[:-1]) / 2
    axs[2].fill_between(r, m.density, color=ORANGE, alpha=0.25)
    glow(axs[2], r, m.density, ORANGE, lw=1.2)
    axs[2].set(title="Radial density", xlabel="Distance from centre of mass (nm)", ylabel="Beads per nm$^3$")
    save_figure(fig, "struct_pairs", **_TXT("struct_pairs"))


def scaling_figure(m, md):
    """Polymer scaling, fractal dimension and structural change."""
    fig, axs = make_fig(1, 3, cell=(4.8, 4.0))
    nu = float(np.nanmean(m.nu))
    axs[0].plot(m.s, m.r_of_s, color=VIOLET, label=f"3D structures (nu = {nu:.2f})")
    k = np.searchsorted(m.s, 5)
    for e, name, c in ((1 / 3, "globule (1/3)", AQUA), (0.5, "ideal chain (1/2)", ORANGE)):
        axs[0].plot(m.s, m.r_of_s[k] * (m.s / m.s[k]) ** e, color=c, ls="--", lw=1, label=name)
    axs[0].set(xscale="log", yscale="log", title="Polymer scaling R(s)", xlabel="Genomic separation s (beads)", ylabel="Mean 3D distance (nm)")
    axs[0].legend(fontsize=8)
    df1, df2 = 1 / m.nu[np.isfinite(m.nu)], m.df_mass[np.isfinite(m.df_mass)]
    bins = np.linspace(min(df1.min(), df2.min()) - 0.1, max(df1.max(), df2.max()) + 0.1, 20)
    axs[1].hist(df1, bins=bins, color=VIOLET, alpha=0.65, edgecolor="white", label=f"1/nu (mean {df1.mean():.2f})")
    axs[1].hist(df2, bins=bins, color=ORANGE, alpha=0.55, edgecolor="white", label=f"mass-radius (mean {df2.mean():.2f})")
    axs[1].set(title="Fractal dimension", xlabel="d_f (3 = space filling)", ylabel="Frames")
    axs[1].legend(fontsize=8)
    if len(m.rmsd_step):
        glow(axs[2], _t(md, len(m.rmsd_step)), m.rmsd_step, AQUA)
        axs[2].axhline(m.rmsd_step.mean(), color=MUTED, ls=":", lw=1)
    axs[2].set(title="Frame-to-frame RMSD", xlabel="MD time (ns)", ylabel="RMSD after alignment (nm)")
    save_figure(fig, "struct_scaling", **_TXT("struct_scaling"))


def compartment_figure_3d(m, md):
    """Where A and B beads sit in 3D (radial position) and how compact each compartment is."""
    c, t = m.compart, _t(md, len(m.compart["r_a"]))
    fig, axs = make_fig(1, 2)
    glow(axs[0], t, c["r_a"], A_RED, label="A (open)")
    glow(axs[0], t, c["r_b"], B_BLUE, label="B (closed)")
    axs[0].set(title="Radial position", xlabel="MD time (ns)", ylabel="Mean distance from centre (nm)")
    axs[0].legend(fontsize=8)
    glow(axs[1], t, c["rg_a"], A_RED, label="A")
    glow(axs[1], t, c["rg_b"], B_BLUE, label="B")
    axs[1].set(title="Compactness of each compartment", xlabel="MD time (ns)", ylabel="Radius of gyration of the subset (nm)")
    axs[1].legend(fontsize=8)
    save_figure(fig, "struct_compartments", **_TXT("struct_compartments"))


# ---- Force-field checks (each figure <= 3 panels): bonds and angles, excluded volume, loop springs ----


def _hist(ax, x, color, bins=50, **kw):
    ax.hist(x, bins=bins, density=True, color=color, alpha=0.55, edgecolor="white", **kw)


def bonds_angles_figure(f, md):
    """Bond-length and bend-angle distributions against their thermal expectation, and angle over time."""
    fig, axs = make_fig(1, 3, cell=(4.8, 4.0))
    _hist(axs[0], f.bond.ravel(), VIOLET)
    x = np.linspace(f.bond.min(), f.bond.max(), 300)
    sd = np.sqrt(f.kT / BOND_K)
    axs[0].plot(x, np.exp(-(x - BOND_LENGTH) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), color=ORANGE, lw=2, label="thermal expectation")
    axs[0].axvline(BOND_LENGTH, color="#111", ls="--", lw=1.2, label="rest length")
    axs[0].set(title="Bond lengths", xlabel="Distance of adjacent beads (nm)", ylabel="Probability density")
    axs[0].legend(fontsize=8)
    th = np.linspace(1, 179, 400)
    boltz = np.sin(np.radians(th)) * np.exp(-f.k_angle * (np.radians(th) - np.pi) ** 2 / (2 * f.kT))     # sin(theta): 3D measure
    boltz /= np.trapezoid(boltz, th)
    _hist(axs[1], f.angle.ravel(), AQUA, bins=60)
    axs[1].plot(th, boltz, color=ORANGE, lw=2, label="angle force, Boltzmann")
    axs[1].set(title="Bend angles (180 = straight)", xlabel="Angle at a bead (degrees)", ylabel="Probability density", xlim=(0, 180))
    axs[1].legend(fontsize=8, loc="upper left")
    t, a = _t(md, len(f.angle)), f.angle
    axs[2].fill_between(t, np.percentile(a, 25, axis=1), np.percentile(a, 75, axis=1), color=AQUA, alpha=0.25, label="25-75%")
    glow(axs[2], t, a.mean(axis=1), AQUA, label="mean")
    axs[2].set(title="Bend angle over time", xlabel="MD time (ns)", ylabel="Angle (degrees)")
    axs[2].legend(fontsize=8)
    save_figure(fig, "struct_bonds_angles", **_TXT("struct_bonds_angles"))


def excluded_volume_figure(f, md):
    """Short-range density of non-bonded pairs, nearest-neighbour distances and overlaps over time."""
    fig, axs = make_fig(1, 3, cell=(4.8, 4.0))
    axs[0].fill_between(f.g_edges, f.g / (f.g[-8:].mean() or 1), color=VIOLET, alpha=0.25)
    glow(axs[0], f.g_edges, f.g / (f.g[-8:].mean() or 1), VIOLET, lw=1.4)
    axs[0].axvline(BOND_LENGTH, color="#111", ls="--", lw=1.2, label="bond length")
    axs[0].set(title="Pair density at short range", xlabel="Distance of non-bonded beads (nm)", ylabel="Density / long-range density")
    axs[0].legend(fontsize=8)
    _hist(axs[1], f.nearest, AQUA, bins=40)
    axs[1].axvline(BOND_LENGTH, color="#111", ls="--", lw=1.2)
    axs[1].set(title="Nearest non-bonded neighbour", xlabel="Distance (nm)", ylabel="Probability density")
    t = _t(md, len(f.overlap))
    glow(axs[2], t, 100 * f.overlap, ORANGE)
    axs[2].set(title="Overlapping pairs over time", xlabel="MD time (ns)", ylabel="Pairs closer than half a bond (%)")
    save_figure(fig, "struct_ev", **_TXT("struct_ev"))


def loop_distance_figure(f, md):
    """Loop-anchor distances (distribution and over time) and the spacing of neighbouring loops."""
    fig, axs = make_fig(1, 3, cell=(4.8, 4.0))
    if len(f.anchor_all):
        _hist(axs[0], f.anchor_all, VIOLET, bins=40)
    axs[0].axvline(f.loop_rest, color="#111", ls="--", lw=1.2, label="spring rest length")
    axs[0].axvline(np.median(f.anchor_all) if len(f.anchor_all) else 0, color=ORANGE, lw=1.6, label="median")
    axs[0].set(title="Loop-anchor distance", xlabel="Distance of the two anchors (nm)", ylabel="Probability density")
    axs[0].legend(fontsize=8)
    t = _t(md, len(f.anchor_mean))
    axs[1].fill_between(t, f.anchor_q[:, 0], f.anchor_q[:, 1], color=VIOLET, alpha=0.25, label="25-75%")
    glow(axs[1], t, f.anchor_mean, VIOLET, label="mean")
    axs[1].set(title="Loop anchors over time", xlabel="MD time (ns)", ylabel="Anchor distance (nm)")
    axs[1].legend(fontsize=8)
    if len(f.gap) > 5:
        edges = np.unique(np.percentile(f.gap, np.linspace(0, 100, 8)))
        idx = np.clip(np.digitize(f.gap, edges) - 1, 0, len(edges) - 2)
        x = np.array([f.gap[idx == i].mean() for i in range(len(edges) - 1) if (idx == i).any()])
        q = np.array([np.percentile(f.dist[idx == i], [25, 50, 75]) for i in range(len(edges) - 1) if (idx == i).any()])
        axs[2].fill_between(x, q[:, 0], q[:, 2], color=AQUA, alpha=0.25, label="25-75%")
        axs[2].plot(x, q[:, 1], color=AQUA, lw=2.2, marker="o", label="median")
        axs[2].legend(fontsize=8)
    axs[2].set(title="Neighbouring loops", xlabel="Genomic gap between them (beads)", ylabel="3D distance of facing anchors (nm)")
    save_figure(fig, "struct_loops", **_TXT("struct_loops"))


def time_figure(m, md):
    """More structural metrics along the trajectory: anisotropy axes, scaling/fractal exponents, local packing."""
    n = len(m.shape["rg"])
    t, tp = _t(md, n), _t(md, n)[m.pair_frames]
    fig, axs = make_fig(1, 3, cell=(4.8, 4.0))
    glow(axs[0], t, m.shape["asphericity"], VIOLET, label="asphericity")
    glow(axs[0], t, m.shape["acylindricity"], ORANGE, lw=1.2, label="acylindricity")
    axs[0].set(title="Shape asymmetry", xlabel="MD time (ns)", ylabel="Gyration-tensor measure (nm$^2$)")
    axs[0].legend(fontsize=8)
    glow(axs[1], tp, m.nu, AQUA, label="scaling exponent $\\nu$")
    glow(axs[1], tp, m.df_mass, VIOLET, lw=1.2, label="fractal dim. (mass-radius)")
    axs[1].axhline(1 / 3, color=MUTED, ls=":", lw=1)
    axs[1].set(title="Scaling and fractal dimension", xlabel="MD time (ns)", ylabel="Dimensionless")
    axs[1].legend(fontsize=8)
    glow(axs[2], tp, m.pairs["contacts"], ORANGE)
    axs[2].set(title="Contacts per bead", xlabel="MD time (ns)", ylabel="Non-bonded contacts (count)")
    save_figure(fig, "struct_time", **_TXT("struct_time"))


# ---- Start / middle / end structures with a filmstrip and the simulation progress bars ----

def _thumb(img, k=4):
    """Block-average an RGB image by k (small copy for the filmstrip)."""
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    return img[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3)).astype(np.uint8)


def _crop_common(images, pad=12):
    """Crop all renders to the same box around the structures (removes the white margin)."""
    ink = np.any([np.any(im < 245, axis=2) for im in images], axis=0)
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    r0, r1 = max(rows[0] - pad, 0), rows[-1] + pad
    c0, c1 = max(cols[0] - pad, 0), cols[-1] + pad
    return [im[r0:r1, c0:c1] for im in images]


def snapshots_figure(images, picks, md, mc):
    """Three large 3D renders (start, middle, end), a filmstrip of all renders, a Monte Carlo step bar and an order-parameter (T) bar.

    images: renders of the MD frames `picks` (evenly spaced, in time order); only the three representative structures
    (A, B, C) are boxed, and the same boxes mark them on the Monte Carlo and T bars.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from viz.theme import apply_theme
    apply_theme()
    fig = plt.figure(figsize=(14.4, 5.4), layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=[7.5, 2.4, 1.0, 1.0])
    step, burn = mc.meta["mc_step"], mc.meta["burnin"]
    snap = np.asarray(md.frame_idx)[picks]                       # MC snapshot behind each render
    steps = snap * step / 1e3
    images = _crop_common(images)
    n = len(images)
    big = [0, n // 2, n - 1]
    for pos, (k, name) in enumerate(zip(big, ("Start", "Middle", "End"))):
        ax = fig.add_subplot(gs[0, pos])
        ax.imshow(images[k])
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"{name}: MC step {int(snap[k] * step):,},  T = {mc.Ts[snap[k]]:.3g}", loc="left", pad=14, fontsize=11)
        ax.grid(False)
        ax.set_frame_on(False)
    x = np.arange(len(mc.Ts)) * step / 1e3                       # MC step axis (thousands)
    strip, mcbar, tbar = (fig.add_subplot(gs[r, :]) for r in (1, 2, 3))
    for ax in (strip, mcbar, tbar):
        ax._twin = True                                          # not a lettered panel
        ax.set(yticks=[], xlim=(0, x[-1]))
        ax.grid(False)
    strip.set(xticklabels=[], ylim=(0, 1.3))
    strip.set_frame_on(False)
    fig.canvas.draw()                                            # resolve the layout to size the thumbnails
    th = _thumb(images[0], 3)
    cell_px = strip.get_window_extent().width / n
    zoom = 0.84 * cell_px * 72 / (fig.dpi * th.shape[1])
    w = x[-1] / n
    for k, img in enumerate(images):                             # one small structure per sampled step, in time order
        xv = w / 2 + k * (x[-1] - w) / max(n - 1, 1)
        strip.add_artist(AnnotationBbox(OffsetImage(_thumb(img, 3), zoom=zoom), (xv, 0.5), frameon=k in big, pad=0.12,
                                        bboxprops=dict(boxstyle="round,pad=0.0,rounding_size=0.25", ec=VIOLET, lw=2.2, fc="white")))
        if k in big:
            strip.text(xv, 1.0, "ABC"[big.index(k)], ha="center", va="bottom", fontsize=8, fontweight="bold", color="white",
                       bbox=dict(boxstyle="round,pad=0.25", fc=VIOLET, ec="none"))
    strip.set_ylabel("Structures\n(in time order)", rotation=0, ha="right", va="center", labelpad=10)
    span = x[-1]
    mcbar.axvspan(0, burn / 1e3, color="#d9d3ea", lw=0)
    mcbar.axvspan(burn / 1e3, span, color=VIOLET, alpha=0.85, lw=0)
    tr = mcbar.get_xaxis_transform()
    if burn / 1e3 > 0.12 * span:
        mcbar.text(burn / 2e3, 0.5, "burn-in", ha="center", va="center", fontsize=8, color=MUTED, transform=tr)
        mcbar.text(burn / 1e3 + 0.02 * span, 0.5, "Monte Carlo sampling", ha="left", va="center", fontsize=8, color="white", transform=tr)
    else:                                                        # burn-in too narrow for its own label
        mcbar.text(burn / 1e3 + 0.01 * span, 0.5, "Monte Carlo sampling   (grey: burn-in)", ha="left", va="center", fontsize=8, color="white", transform=tr)
    mcbar.set(xticklabels=[])
    mcbar.set_ylabel("Monte Carlo", rotation=0, ha="right", va="center", labelpad=10)
    mcbar.scatter(steps[big], np.ones(3), c="#111", marker="v", s=46, ec="white", lw=0.8, transform=tr, clip_on=False, zorder=5)
    T = np.asarray(mc.Ts, dtype=float)
    tbar.imshow(T[None, :], aspect="auto", cmap="YlOrRd", norm=Normalize(T.min(), T.max() if T.max() > T.min() else T.min() + 1),
                extent=(0, span, 0, 1))
    tbar.scatter(steps[big], np.zeros(3), c="#111", marker="^", s=46, ec="white", lw=0.8, transform=tbar.get_xaxis_transform(), clip_on=False, zorder=5)
    tbar.set(xlabel="MC step (thousands)")
    tbar.set_ylabel("Order parameter\n(temperature T)", rotation=0, ha="right", va="center", labelpad=10)
    tbar.text(0.995, 0.5, f"T: {T[0]:.3g} \u2192 {T[-1]:.3g}" if abs(T[0] - T[-1]) > 1e-9 else f"T = {T[0]:.3g} (constant)",
              ha="right", va="center", fontsize=8, color="black", transform=tbar.transAxes)
    save_figure(fig, "struct_snapshots", **_TXT("struct_snapshots"))
