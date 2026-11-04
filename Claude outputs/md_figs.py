"""Figures of the molecular dynamics stage and MC-vs-MD comparison."""
import numpy as np

from helpers.matrices import minmax, pearson_maps, ps_curve
from viz.panels import heatmap
from viz.theme import AQUA, ORANGE, VIOLET, make_fig, make_mosaic, save_figure


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
    axs[0].plot(t, md.rg, color=VIOLET)
    axs[0].set(title="Radius of gyration", xlabel="MD time (ns)", ylabel="R_g (nm)")
    axs[1].plot(t, md.epot, color=ORANGE)
    axs[1].set(title="Potential energy", xlabel="MD time (ns)", ylabel="Energy (kJ/mol)")
    save_figure(fig, "md_series")


def _norm_ps(mat):
    """P(s) scaled to its value at s = 1 (the diagonal itself is trivially 1)."""
    p = ps_curve(mat)[1:]
    return p / p[0] if p[0] > 0 else p


def mc_vs_md(mc_map, md_map, geom):
    """Loop-ensemble map vs 3D map, and their decay with separation."""
    r = pearson_maps(mc_map, md_map)
    fig, ax = make_mosaic([["mc", "md", "ps"]], size=(15, 4.6))
    heatmap(fig, ax["mc"], minmax(np.log1p(mc_map)), geom, "Loop ensemble (MC)", "Normalised contact (0-1)", vmax=1)
    heatmap(fig, ax["md"], minmax(np.log1p(md_map)), geom, f"3D structures (MD), r = {r:.2f}", "Normalised contact (0-1)", vmax=1)
    s = np.arange(1, len(mc_map))
    ax["ps"].loglog(s, _norm_ps(mc_map), color=VIOLET, label="MC loop ensemble")
    ax["ps"].loglog(s, _norm_ps(md_map), color=AQUA, label="MD 3D structures")
    ax["ps"].set(title="Contact decay P(s)", xlabel="Separation s (beads)", ylabel="P(s) / P(1)")
    ax["ps"].legend()
    save_figure(fig, "mc_vs_md")
