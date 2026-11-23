"""Figures comparing independent replicas: mean and spread of the maps, and per-replica MD traces (<= 3 panels each)."""
import numpy as np

from loopsage.analysis.replicas import map_correlation, map_spread
from loopsage.common.utils import moving_average
from loopsage.viz.theme import FREE, HEAT, heatmap, make_fig, make_mosaic, save_figure

COLORS = ["#6a3de8", "#111111", "#1baf7a", "#a98df0", "#4b2bb8", "#e0457b", "#f58a3c", "#2b62d9"]


def _corr_panel(fig, ax, corr):
    """Replica-vs-replica correlation matrix (numbers when there are few replicas)."""
    n = len(corr)
    im = ax.imshow(corr, cmap=HEAT, vmin=min(0.0, float(np.nanmin(corr))), vmax=1)
    ax.grid(False)
    ax.set(title="Map correlation", xlabel="Replica", ylabel="Replica", xticks=range(n), yticks=range(n))
    if n <= 8:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#111" if corr[i, j] < 0.75 else "#fff")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")


def replica_maps_figure(key, label, maps, geom, unit, section, cmap=None):
    """Mean map, spread between replicas (std) and the pairwise correlation of the maps."""
    mean, std = map_spread(maps)
    fig, ax = make_mosaic([["mean", "std", "corr"]], size=(16, 4.6), width_ratios=[1, 1, 0.9])
    heatmap(fig, ax["mean"], mean, geom, f"Mean {label}", unit, cmap=cmap)
    heatmap(fig, ax["std"], std, geom, "Std between replicas", f"Std of {unit.lower()}", cmap=cmap)
    ax["std"].set_ylabel("")
    _corr_panel(fig, ax["corr"], map_correlation(maps))
    save_figure(fig, key, section=section)


def md_distance_figure(avg_dists, geom):
    """Mean 3D distance map over replicas, its spread and the relative spread (std / mean)."""
    mean, std = map_spread(avg_dists)
    fig, axs = make_fig(1, 3, cell=(5.0, 4.4))
    heatmap(fig, axs[0], mean, geom, "Mean 3D distance", "Distance (nm)", cmap=FREE)
    heatmap(fig, axs[1], std, geom, "Std between replicas", "Std of distance (nm)", cmap=FREE)
    heatmap(fig, axs[2], std / np.maximum(mean, 1e-9), geom, "Relative spread", "Std / mean", cmap=FREE)
    for a in axs[1:]:
        a.set_ylabel("")
    save_figure(fig, "md_replica_distances", section="molecular_dynamics")


def md_traces_figure(md):
    """Radius of gyration and potential energy along the MD frames, one line per replica."""
    fig, axs = make_fig(1, 2)
    for k, c in enumerate(md.chains):
        t = (np.arange(len(c.rg)) + 1) * c.meta["ps_per_frame"] / 1000.0
        axs[0].plot(t, moving_average(c.rg, max(1, len(c.rg) // 25)), color=COLORS[k % 8], lw=1.5, label=f"replica {k}")
        axs[1].plot(t, moving_average(c.epot, max(1, len(c.epot) // 25)), color=COLORS[k % 8], lw=1.5)
    axs[0].set(title="Radius of gyration", xlabel="MD time (ns)", ylabel="R_g (nm)")
    axs[1].set(title="Potential energy", xlabel="MD time (ns)", ylabel="Energy (kJ/mol)")
    if len(md.chains) <= 6:
        axs[0].legend(fontsize=8)
    save_figure(fig, "md_replicas", section="molecular_dynamics")
