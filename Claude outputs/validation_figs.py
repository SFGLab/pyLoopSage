"""Figures comparing simulation with experiment, and linking numbers."""
import numpy as np

from helpers.matrices import minmax
from viz.theme import AQUA, EXP_COLOR, MUTED, SIM_COLOR, VIOLET, make_fig, make_mosaic, save_figure

_TITLES = {"pearson": "Loop strength", "pearson_direct": "Per-loop strength",
           "ps_decay": "Contact decay P(s)", "insulation": "Boundary proxy"}


def signal_figure(key, signal, xlabel, c):
    """Experimental trace, simulated trace, and their correlation (3 panels)."""
    fig, ax = make_mosaic([["exp", "sc"], ["sim", "sc"]], size=(11, 5.4), width_ratios=[2, 1.1])
    i = np.arange(len(c["exp"]))
    for a, v, col, lab in ((ax["exp"], c["exp"], EXP_COLOR, "Experimental (z-score)"),
                           (ax["sim"], c["sim"], SIM_COLOR, "Simulated (z-score)")):
        a.fill_between(i, v, color=col, alpha=0.15)
        a.plot(i, v, color=col)
        a.set_ylabel(lab)
    ax["exp"].set_title(f"{_TITLES.get(key, signal)}: experiment vs simulation")
    ax["exp"].tick_params(labelbottom=False)
    ax["sim"].set_xlabel(xlabel)
    e, s = minmax(c["exp"]), minmax(c["sim"])
    ax["sc"].scatter(e, s, s=16, color=VIOLET, alpha=0.55, edgecolor="white", linewidth=0.3)
    ax["sc"].plot([0, 1], [0, 1], color=MUTED, ls=":", label="y = x")
    if np.std(e) > 0:
        ax["sc"].plot([0, 1], np.polyval(np.polyfit(e, s, 1), [0, 1]), color=AQUA, label="best fit")
    ax["sc"].set(title="Correlation", xlabel="Experimental (0-1)", ylabel="Simulated (0-1)", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax["sc"].text(0.05, 0.95, f"Pearson r = {c['pearson']:.3f}\nSpearman rho = {c['spearman']:.3f}", va="top",
                  transform=ax["sc"].transAxes, fontsize=9)
    ax["sc"].legend(loc="lower right")
    save_figure(fig, f"validation_{key}", title=f"Validation: {_TITLES.get(key, signal)}",
                caption=f"Experimental versus simulated {signal} (n = {c['n']}).")


def validation_summary(rows):
    """Pearson r with 95% confidence intervals for every comparison."""
    rows = [r for r in rows if np.isfinite(r["ci_lo"])]
    if not rows:
        return
    fig, axs = make_fig(1, 1, cell=(8, 0.7 * len(rows) + 2.2))
    y = np.arange(len(rows))[::-1]
    r = np.array([x["pearson"] for x in rows])
    lo, hi = r - np.array([x["ci_lo"] for x in rows]), np.array([x["ci_hi"] for x in rows]) - r
    axs[0].errorbar(r, y, xerr=[lo, hi], fmt="o", color=VIOLET, ecolor=VIOLET, capsize=3)
    axs[0].axvline(0, color=MUTED, lw=0.8)
    axs[0].set_yticks(y, [f"{x['label']}\n{x['signal']}" for x in rows])
    axs[0].set(title="Agreement with experiment", xlabel="Pearson r (95% CI)", xlim=(-1, 1))
    save_figure(fig, "validation_summary")
