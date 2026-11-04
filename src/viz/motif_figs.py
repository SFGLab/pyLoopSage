"""CTCF motif search diagnostics: loops kept, motif scores against the cutoff, orientation of loop anchors."""
import numpy as np

from viz.theme import INK, make_fig, MUTED, SEQ, save_figure, VIOLET


def search_figure(res):
    """Loop funnel, best motif score per anchor against the cutoff, left vs right anchor orientation."""
    st, a, lo = res.stats, res.anchors, res.loops
    fig, axs = make_fig(1, 3, cell=(5.0, 4.0))
    steps = [("in file", st["n_file"]), ("in region", st["n_region"]), ("kept (PET)", st["n_kept"]),
             ("one anchor oriented", st["n_both"] + st["n_one"]), ("both oriented", st["n_both"])]
    names, vals = zip(*steps)
    y = np.arange(len(vals))[::-1]
    axs[0].barh(y, vals, color=VIOLET, alpha=0.85, edgecolor="white")
    for yi, v in zip(y, vals):
        axs[0].text(v, yi, f" {v}", va="center", fontsize=9, color=INK)
    axs[0].set_yticks(y, names)
    axs[0].set(title="Loops after each step", xlabel="Loops", xlim=(0, max(vals) * 1.18))
    hit = a[a["n_hits"] > 0]
    if len(hit):
        top = max(st["max_score"], hit["best_score"].max()) + 0.01
        axs[1].hist(hit["best_score"], bins=24, range=(st["cutoff"], top), color=VIOLET, alpha=0.85, edgecolor="white")
    axs[1].axvline(st["cutoff"], color=INK, ls="--", lw=1.4, label=f"cutoff (p = {st['pvalue']:g})")
    axs[1].axvline(st["max_score"], color=MUTED, ls=":", lw=1.4, label="best possible")
    axs[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    axs[1].set(title=f"Best motif score per anchor ({len(hit)}/{len(a)} with a hit)", xlabel="Log-odds score", ylabel="Anchors")
    both = (lo["p1"] >= 0) & (lo["p2"] >= 0)
    if both.sum() > 3:
        left = lo["start1"] <= lo["start2"]
        pl, pr = np.where(left, lo["p1"], lo["p2"])[both], np.where(left, lo["p2"], lo["p1"])[both]
        h = axs[2].hist2d(pl, pr, bins=10, range=[[0, 1], [0, 1]], cmap=SEQ)
        axs[2].text(0.14, 0.95, "convergent", color=INK, fontsize=9, va="center", ha="left")
        axs[2].set(title="Left vs right anchor orientation", xlabel="P(reverse), left anchor", ylabel="P(reverse), right anchor")
        fig.colorbar(h[3], ax=axs[2], label="Loops", shrink=0.85)
    else:
        axs[2].text(0.5, 0.5, "fewer than 4 loops with\nboth anchors oriented", ha="center", va="center", color=MUTED, transform=axs[2].transAxes)
        axs[2].axis("off")
    save_figure(fig, "motif_search")
