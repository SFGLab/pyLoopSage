"""Figures of the molecular dynamics stage and MC-vs-MD comparison.

- Contact map next to its PC1, insulation and P(s) features.
- Features (PC1, insulation, P(s)) of a contact map, drawn next to the map.
"""
import numpy as np

from loopsage.analysis.maps import compare_features, map_features, minmax, pearson_maps, ps_curve
from loopsage.analysis.structure import anchor_distances, distance_scaling
from loopsage.viz.theme import (
    A_RED,
    AQUA,
    B_BLUE,
    EXP_COLOR,
    genomic_x,
    glow,
    heatmap,
    make_fig,
    make_mosaic,
    MUTED,
    ORANGE,
    save_figure,
    SIM_COLOR,
    VIOLET,
)


# ---- Figures of the molecular dynamics stage and MC-vs-MD comparison -------

def md_maps(md, geom):
    """Mean and variability of the MD contact map."""
    fig, axs = make_fig(1, 2, cell=(5.6, 4.8))
    heatmap(fig, axs[0], md.avg_map, geom, "Mean contact map", "Contact probability (0-1)", vmax=1)
    heatmap(fig, axs[1], md.std_map, geom, "Frame-to-frame std", "Std of contact probability")
    save_figure(fig, "md_maps")


def md_series(md):
    """Radius of gyration and potential energy along the MD frames."""
    t = (np.arange(len(md.rg)) + 1) * md.meta["ps_per_frame"] / 1000.0
    fig, axs = make_fig(1, 2)
    glow(axs[0], t, md.rg, VIOLET)
    axs[0].set(title="Radius of gyration", xlabel="MD time (ns)", ylabel="R_g (nm)")
    glow(axs[1], t, md.epot, ORANGE)
    axs[1].set(title="Potential energy", xlabel="MD time (ns)", ylabel="Energy (kJ/mol)")
    save_figure(fig, "md_series")


def _norm_ps(mat):
    """P(s) scaled to its value at s = 1 (the diagonal itself is trivially 1)."""
    p = ps_curve(mat)[1:]
    return p / p[0] if p[0] > 0 else p


def mc_vs_md(mc_map, md_map, geom):
    """Loop-ensemble map vs 3D map (shared colour scale), and their P(s) decay."""
    r = pearson_maps(mc_map, md_map)
    fig, ax = make_mosaic([["mc", "md", "ps"]], size=(15.5, 4.6), width_ratios=[1, 1, 1.15])
    a = heatmap(fig, ax["mc"], minmax(np.log1p(mc_map)), geom, "Loop ensemble (MC)", "", vmax=1, cbar=False, gamma=0.6)
    heatmap(fig, ax["md"], minmax(np.log1p(md_map)), geom, f"3D structures (MD), r = {r:.2f}", "", vmax=1, cbar=False, gamma=0.6)
    ax["md"].set_ylabel("")
    cb = fig.colorbar(a, ax=[ax["mc"], ax["md"]], shrink=0.8, pad=0.02)
    cb.set_label("Normalised contact (0-1)")
    cb.outline.set_visible(False)
    for key, m, c, name in (("mc", mc_map, VIOLET, "MC loop ensemble"), ("md", md_map, AQUA, "MD 3D structures")):
        p = _norm_ps(m)
        s = np.arange(1, len(p) + 1)
        ok = p > 1e-8                                   # drop empty separations on the log axis
        glow(ax["ps"], s[ok], p[ok], c, label=name)
    ax["ps"].set(xscale="log", yscale="log", title="Contact decay P(s)", xlabel="Separation s (beads)", ylabel="P(s) / P(1)")
    ax["ps"].legend()
    save_figure(fig, "mc_vs_md")


def md_geometry(md, mc, geom):
    """Polymer scaling R(s) and spatial proximity of loop anchors."""
    s, r, nu = distance_scaling(md.coords)
    loop, ctrl = anchor_distances(md, mc)
    fig, axs = make_fig(1, 2)
    glow(axs[0], s, r, VIOLET, label=f"3D structures (nu = {nu:.2f})")
    for exp, name, c in ((1 / 3, "compact globule (1/3)", AQUA), (0.5, "ideal chain (1/2)", ORANGE)):
        k = np.searchsorted(s, 5)
        axs[0].plot(s, r[k] * (s / s[k]) ** exp, color=c, ls="--", lw=1, label=name)
    axs[0].set(xscale="log", yscale="log", title="Polymer scaling", xlabel="Genomic separation s (beads)", ylabel="Mean 3D distance R(s) (nm)")
    axs[0].legend(fontsize=8)
    hi = np.percentile(np.concatenate([loop, ctrl]), 99)
    bins = np.linspace(0, hi, 40)
    axs[1].hist(ctrl, bins=bins, color=ORANGE, alpha=0.45, density=True, edgecolor="white", label=f"Random pairs, same s (median {np.median(ctrl):.1f} nm)")
    axs[1].hist(loop, bins=bins, color=VIOLET, alpha=0.7, density=True, edgecolor="white", label=f"Loop anchors (median {np.median(loop):.1f} nm)")
    axs[1].set(title="Are loop anchors close in 3D?", xlabel="Anchor-anchor distance (nm)", ylabel="Probability density (1/nm)")
    axs[1].legend(fontsize=8)
    save_figure(fig, "md_geometry")
    return dict(nu=nu, loop_nm=float(np.median(loop)), ctrl_nm=float(np.median(ctrl)))


# ---- Contact map next to its PC1, insulation and P(s) features -------------



def _ins_panel(ax, x, f, color):
    ax.plot(x, f.insulation, color=color, lw=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    b = f.boundaries
    ax.plot(x[b], f.insulation[b], "v", color="#111", ms=6, label=f"{len(b)} boundaries")
    ax.legend(loc="upper right")


def _ps_panel(ax, f, geom, color, label=None):
    kb = f.s * geom.bp_per_bead / 1e3
    ax.loglog(kb, f.ps, color=color, lw=2, label=label)


def map_features_figure(label, mat, f, geom, key, color=VIOLET, ref=None):
    """Heat map (left) with insulation score and P(s) decay on the right."""
    x = geom.axis_mb()
    fig, ax = make_mosaic([["h", "ins"], ["h", "ps"]], size=(11.5, 6.2), width_ratios=[1.05, 1.0])
    heatmap(fig, ax["h"], minmax(np.log1p(mat)), geom, f"{label} contact map", "Normalised contact (0-1)", vmax=1, gamma=0.6)
    ax["h"].plot(x[f.boundaries], x[f.boundaries], "v", color="#111", ms=5)       # boundaries on the diagonal
    _ins_panel(ax["ins"], x, f, color)
    ax["ins"].set(title="Insulation score (low = boundary)", ylabel="log2 insulation")
    genomic_x(ax["ins"], geom)
    _ps_panel(ax["ps"], f, geom, color)
    ax["ps"].set(title=f"Diagonal decay P(s), slope {f.slope:.2f}", xlabel="Separation s (kb)", ylabel="Mean contact")
    save_figure(fig, key, title=f"{label}: map, insulation and decay", section="molecular_dynamics" if "md" in key else "monte_carlo",
                caption=f"{label} contact map with its insulation score (TAD-like boundaries) and diagonal decay P(s) next to it.")


def features_compare_figure(mc, md, geom, ref=None, exp=None, mc_pc1=None):
    """MC, MD and (when available) experimental features: PC1 vs the input compartments, insulation score and P(s) decay."""
    c = compare_features(mc, md)
    x = geom.axis_mb()
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    fig, axs = make_fig(1, 3, cell=(5.2, 4.2))
    c["pc1"] = abs(r(mc_pc1, md.pc1)) if mc_pc1 is not None else float("nan")     # MC compartments = average epigenetic state (none without the model)
    sgn = np.sign(r(mc_pc1, md.pc1) or 1) if mc_pc1 is not None else 1
    note = lambda v: f", r = {v:.2f}" if np.isfinite(v) else ""
    if ref is not None:
        ref = np.asarray(ref, float)
        axs[0].plot(x, ref / (np.abs(ref).max() or 1), color="#111", lw=1.6, ls="--", label="input compartments")
    if mc_pc1 is not None:
        axs[0].plot(x, mc_pc1, color=VIOLET, lw=2, label="MC epigenetic state" + note(r(mc_pc1, ref) if ref is not None else np.nan))
    axs[0].plot(x, md.pc1 * sgn, color=AQUA, lw=2, label="MD 3D" + note(r(md.pc1 * sgn, ref) if ref is not None else np.nan))
    axs[0].set(title="Compartments" + (f", MC vs MD |r| = {c['pc1']:.2f}" if mc_pc1 is not None else ""), ylabel="Compartment signal (scaled)")
    genomic_x(axs[0], geom)
    axs[0].legend(fontsize=8)
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) or 1.0)           # z-scores: the maps have different contrast
    ok = ~np.isnan(mc.insulation) & ~np.isnan(md.insulation) & (~np.isnan(exp.insulation) if exp is not None else True)
    if exp is not None:
        axs[1].plot(x, z(exp.insulation), color="#111", lw=1.6, ls="--", label="experiment (loop map)")
    axs[1].plot(x, z(mc.insulation), color=VIOLET, lw=2, label="MC loops" + note(r(mc.insulation[ok], exp.insulation[ok]) if exp is not None else np.nan))
    axs[1].plot(x, z(md.insulation), color=AQUA, lw=2, label="MD 3D" + note(r(md.insulation[ok], exp.insulation[ok]) if exp is not None else np.nan))
    axs[1].set(title=f"Insulation score, MC vs MD r = {c['insulation']:.2f}", ylabel="Insulation (z-score)")
    genomic_x(axs[1], geom)
    axs[1].legend(fontsize=8)
    for f, col, name in ((exp, "#111", "experiment"), (mc, VIOLET, "MC loops"), (md, AQUA, "MD 3D")):
        if f is None:
            continue
        m = f.ps > 0
        lab = f"{name}, slope {f.slope:.2f}"
        if exp is not None and f is not exp:
            both = m & (exp.ps > 0)
            lab += note(r(np.log(f.ps[both]), np.log(exp.ps[both])))
        axs[2].plot(f.s[m] * geom.bp_per_bead / 1e3, f.ps[m] / f.ps[m][0], color=col, lw=2.2, ls="--" if f is exp else "-", label=lab)
    axs[2].set(xscale="log", yscale="log", title=f"P(s) decay, MC vs MD r = {c['ps']:.2f}", xlabel="Separation s (kb)", ylabel="P(s) / P(1)")
    axs[2].legend(fontsize=8)
    save_figure(fig, "features_compare", title="MC, MD and experiment: features compared", section="molecular_dynamics",
                caption="PC1 (against the input compartments), insulation score and diagonal decay P(s) from the loop ensemble (MC), the 3D structures (MD) and the experiment.")
    return c


# ---- Figures comparing simulation with experiment, and linking numbers -----

# ---- Features (PC1, insulation, P(s)) of a contact map, drawn next to the map ----

def map_feature_report(ctx, name, mat, label):
    """Compute and plot the features; kept on the context for the MC-vs-MD comparison."""
    ref = ctx.inputs.comp
    f = map_features(mat, ref)
    ctx.features = {**(ctx.features or {}), name: f}
    map_features_figure(label, mat, f, ctx.geom, f"{name}_features", ref=ref)
