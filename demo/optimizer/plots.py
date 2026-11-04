"""Figures of the 1000-bead optimiser benchmark (reads results.json and maps.npz written by bench1000.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bench1000 as b

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
R = json.load(open(Path(__file__).resolve().parent / "results.json"))
res, traces = R["results"], R["traces"]
maps = np.load(Path(__file__).resolve().parent / "maps.npz")
COL = {"default parameters": "#7b7596", "random search (MC)": "#6a3de8", "Nelder-Mead (MC)": "#1baf7a",
       "A: ideal gas (Boltzmann)": "#f2a900", "A+C: ideal gas + reweighting": "#e5307f", "oracle (true parameters and sites)": "#111111"}
METHODS = [m for m in COL if m in res and m != "oracle (true parameters and sites)"]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})


def clean(ax):
    ax.grid(alpha=0.25)


def convergence():
    """Best loss so far against wall-clock time, and time needed to reach 95% of each method's own improvement."""
    fig, (a, c) = plt.subplots(1, 2, figsize=(12.5, 4.4), layout="constrained")
    t95 = {}
    for m, tr in traces.items():
        t, best = np.array(tr).T
        a.step(np.r_[t, t[-1] * 1.0], np.r_[best, best[-1]], where="post", color=COL[m], lw=2, label=m)
        goal = best[0] - 0.95 * (best[0] - best[-1])
        t95[m] = float(t[np.flatnonzero(best <= goal)[0]])
    a.axhline(res["default parameters"]["loss"], color=COL["default parameters"], ls="--", lw=1.5, label="default parameters (final run)")
    a.axhline(res["oracle (true parameters and sites)"]["loss"], color="#111", ls=":", lw=1.5, label="oracle (true parameters)")
    a.set(xscale="log", xlabel="Wall-clock time (s)", ylabel="Best loss so far (cheap MC evaluations)", title="Convergence")
    a.legend(fontsize=8, frameon=False, loc="upper right")
    clean(a)
    x = np.arange(len(traces))
    tot = [res[m]["seconds"] for m in traces]
    c.bar(x - 0.2, tot, 0.4, color=[COL[m] for m in traces], label="total optimisation time")
    c.bar(x + 0.2, [t95[m] for m in traces], 0.4, color=[COL[m] for m in traces], alpha=0.45, hatch="//", label="time to 95% of own improvement")
    for xi, v in zip(x - 0.2, tot):
        c.text(xi, v + 1, f"{v:.0f} s", ha="center", fontsize=8)
    c.set_xticks(x, [m.replace(": ", ":\n").replace(" + ", "\n+ ").replace(" (", "\n(") for m in traces], fontsize=8)
    c.set(ylabel="Seconds", title="Time to optimise")
    c.legend(fontsize=8, frameon=False)
    clean(c)
    fig.savefig(OUT / "fig1_convergence.png", dpi=170)
    return t95


def quality():
    """Four scores per method, insulation profiles and decay curves against the artificial Hi-C (3 panels)."""
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    keys = [("insulation_r", "Insulation\nscore r"), ("decay_r", "Decay P(s)\nr (log)"), ("oe_r", "Pixel r\n(O/E, decay removed)"), ("diagonal_r", "Per-diagonal r\n(s = 2..300)")]
    ax, w = axs[0], 0.8 / len(METHODS)
    for i, m in enumerate(METHODS + ["oracle (true parameters and sites)"]):
        ax.bar(np.arange(4) + (i - 3) * 0.8 / 6, [res[m][k] for k, _ in keys], 0.8 / 6, color=COL[m], label=m.split(" (true")[0])
    ax.set_xticks(range(4), [l for _, l in keys], fontsize=8)
    ax.set(ylim=(0.4, 1.02), ylabel="Pearson correlation with the artificial Hi-C", title="Agreement with Hi-C (higher = better)")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    clean(ax)
    hic = maps["Hi-C"]
    best = min(METHODS[1:], key=lambda m: res[m]["loss"])
    show = {"Hi-C": ("#111111", hic), "default parameters": (COL["default parameters"], maps["default_parameters"]), best: (COL[best], maps[best.replace(" ", "_")])}
    for name, (c, M) in show.items():
        M = M / M[b._UP].mean()
        ps, ins = b.features(M)
        axs[1].plot(np.arange(b.W, b.N - b.W), np.log2(ins), color=c, lw=1.2, label=name)
        axs[2].loglog(np.arange(1, b.N), ps, color=c, lw=2, label=name)
    axs[1].set(xlabel="Bead", ylabel="log2 insulation score", title="Insulation score")
    axs[2].set(xlabel="Separation s (beads)", ylabel="P(s) (mean contact)", title="Contact decay")
    for a in axs[1:]:
        a.legend(fontsize=8, frameon=False)
        clean(a)
    fig.savefig(OUT / "fig2_quality.png", dpi=170)
    return best


def heatmaps(best):
    """Artificial Hi-C, default-parameter map and best fitted map (zoom on beads 200-500, upper triangle |i-j| shown fully)."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    sl = slice(200, 500)
    for ax, (name, M) in zip(axs, [("Artificial Hi-C", maps["Hi-C"]), ("Default parameters (f = b = 1)", maps["default_parameters"]), (f"Fitted: {best}", maps[best.replace(" ", "_")])]):
        M = M / M[b._UP].mean()
        im = ax.imshow(np.log10(M[sl, sl] + 0.02), cmap="Reds", origin="upper", vmin=np.log10(0.05), vmax=np.log10(np.percentile(M[sl, sl], 99.5)))
        ax.set(title=name, xticks=[], yticks=[])
        ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.8, label="log10 contact (normalised)")
    fig.savefig(OUT / "fig3_maps.png", dpi=170)


if __name__ == "__main__":
    t95 = convergence()
    best = quality()
    heatmaps(best)
    json.dump(t95, open(OUT.parent / "t95.json", "w"), indent=1)
    print("best:", best, t95)
