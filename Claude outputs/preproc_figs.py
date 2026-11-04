"""Figures of the preprocessed input data."""
import numpy as np

from viz.panels import genomic_x, heatmap
from reporting.explanations import EXPLAIN
from viz.theme import AQUA, ORANGE, VIOLET, make_fig, save_figure


def binding_profiles(inputs, geom):
    """L/R anchor profiles along the region (+ loop-length histogram for loop files)."""
    lo, x = inputs.loops, geom.axis_mb()
    has_loops = len(lo.loop_lengths) > 0
    fig, axs = make_fig(1, 2 if has_loops else 1)
    axs[0].plot(x, lo.L, color=VIOLET, label="L (left anchor)")
    axs[0].plot(x, lo.R, color=ORANGE, label="R (right anchor)")
    axs[0].set_title("Anchor binding profiles")
    axs[0].set_ylabel("Signal (scaled, a.u.)")
    genomic_x(axs[0], geom)
    axs[0].legend()
    if has_loops:
        kb = geom.beads_to_bp(lo.loop_lengths) / 1e3
        axs[1].hist(kb, bins=25, color=VIOLET, alpha=0.75, edgecolor="white")
        axs[1].axvline(kb.mean(), color=ORANGE, ls="--", label=f"mean {kb.mean():.0f} kb")
        axs[1].set(title="Loop lengths in the input", xlabel="Loop length (kb)", ylabel="Number of loops")
        axs[1].legend()
    save_figure(fig, "binding_profiles")


def loop_matrices(inputs, geom):
    """Binary loop adjacency J and weighted loop map J_loss."""
    lo = inputs.loops
    fig, axs = make_fig(1, 2, cell=(5.6, 4.6))
    heatmap(fig, axs[0], lo.J, geom, "Loop adjacency J", "1 = anchor pair / chain neighbour")
    heatmap(fig, axs[1], lo.J_loss, geom, "Weighted loop map J_loss", "Loop strength (a.u.)",
            diverging=lo.J_loss.min() < 0)
    save_figure(fig, "loop_matrices")


def track_figures(inputs, geom):
    """Extra tracks (BigWig, LEF loading, compartments), at most 3 per figure."""
    tracks = [(n, v) for n, v in zip(inputs.bw_names, inputs.BWs)]
    if inputs.lef_track is not None:
        tracks.append(("LEF loading preference", inputs.lef_track))
    if inputs.comp is not None:
        tracks.append(("Compartment signal (A > 0, B < 0)", inputs.comp))
    colors = [VIOLET, ORANGE, AQUA]
    for part, i in enumerate(range(0, len(tracks), 3), 1):
        chunk = tracks[i:i + 3]
        fig, axs = make_fig(len(chunk), 1, cell=(9.5, 2.4), sharex=True)
        for k, (ax, (name, v)) in enumerate(zip(axs, chunk)):
            ax.fill_between(geom.axis_mb(), v, color=colors[k], alpha=0.35)
            ax.plot(geom.axis_mb(), v, color=colors[k], lw=1)
            ax.set_title(name)
            ax.set_ylabel("Signal (a.u.)")
        genomic_x(axs[-1], geom)
        save_figure(fig, f"tracks_{part}", title=f"Input tracks ({part})", section="input",
                    caption=EXPLAIN["tracks"]["caption"], explain=EXPLAIN["tracks"]["explain"])
