"""Figure of the parameter optimizer: loss of every evaluation, and the parameters it visited (2 panels)."""
import numpy as np

from reporting.explanations import EXPLAIN
from viz.theme import AQUA, make_fig, MUTED, ORANGE, save_figure, VIOLET


def convergence_figure(res):
    """Loss and best-so-far against evaluation number, and the path of K, f and b (normalised to the start)."""
    tr = np.array(res.trace)
    k = np.arange(1, len(tr) + 1)
    fig, axs = make_fig(1, 2, cell=(5.8, 4.2))
    axs[0].scatter(k, tr[:, 1], s=18, color=MUTED, alpha=0.7, label="evaluation")
    axs[0].step(k, np.minimum.accumulate(tr[:, 1]), where="post", color=VIOLET, lw=2, label="best so far")
    axs[0].axhline(res.start["loss"], color=ORANGE, ls="--", lw=1.4, label="start (long check run)")
    axs[0].axhline(res.best["loss"], color=VIOLET, ls=":", lw=1.4, label="chosen (long check run)")
    axs[0].set(title="Loss", xlabel="Evaluation (short MC run)", ylabel="Loss (lower = better)")
    axs[0].legend(fontsize=8)
    for col, name, c in ((2, "K", VIOLET), (3, "f", AQUA), (4, "b", ORANGE), (5, "T scale", "#e5307f")):
        axs[1].plot(k, tr[:, col] / tr[0, col], color=c, lw=1.6, label=name)
    axs[1].set(title="Parameters visited", xlabel="Evaluation", ylabel="Value / start value", yscale="log")
    axs[1].legend(fontsize=8)
    save_figure(fig, "optimizer", title=EXPLAIN["optimizer"]["title"], caption=EXPLAIN["optimizer"]["caption"],
                explain=EXPLAIN["optimizer"]["explain"], section="optimizer")


def landscape_figure(res):
    """Loss against K, f and b for every evaluation (colour = order, size = temperature scale); the star is the chosen point."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    tr = np.array(res.trace)
    cmap = LinearSegmentedColormap.from_list("order", ["#d9d3ea", "#6a3de8", "#111111"])
    fig, axs = make_fig(1, 3, cell=(4.8, 4.0))
    size = 25 + 45 * (tr[:, 5] - tr[:, 5].min()) / (np.ptp(tr[:, 5]) or 1.0)
    for ax, col, name, log in zip(axs, (2, 3, 4), ("K (LEFs)", "f (folding)", "b (binding)"), (False, True, True)):
        sc = ax.scatter(tr[:, col], tr[:, 1], c=np.arange(len(tr)), cmap=cmap, s=size, edgecolor="white", lw=0.5)
        key = ("K", "f", "b")[col - 2]
        ax.scatter([res.best[key]], [res.best["loss"]], marker="*", s=240, color="#e5307f", edgecolor="white", lw=0.8, zorder=5, label="chosen")
        ax.set(xlabel=name, ylabel="Loss (short run)", title=name.split(" ")[0], xscale="log" if log else "linear")
        if log:
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8)
    save_figure(fig, "optimizer_landscape", title=EXPLAIN["optimizer_landscape"]["title"], caption=EXPLAIN["optimizer_landscape"]["caption"],
                explain=EXPLAIN["optimizer_landscape"]["explain"], section="optimizer")
