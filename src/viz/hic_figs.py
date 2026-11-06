"""Hi-C figures: the input map, the simulated maps against it, and the scorecard."""
import numpy as np

from validation.hic_compare import METRICS
from viz.theme import EXP_COLOR, genomic_x, heatmap, make_fig, MUTED, save_figure, SIM_COLOR, VIOLET


def _norm(m):
    """Map scaled to its 99th percentile, clipped to [0, 1] (diagonal blank)."""
    m = np.array(m, float)
    np.fill_diagonal(m, 0)
    v = np.nanpercentile(m[m > 0], 99) if (m > 0).any() else 1.0
    return np.clip(np.nan_to_num(m) / v, 0, 1)


def _decay(ax, geom, data, model=None):
    """P(s) on log-log axes (distance in kb)."""
    s = np.arange(1, len(data) + 1) * geom.bp_per_bead / 1e3
    ax.loglog(s, data, color=EXP_COLOR, lw=2, label="Hi-C")
    if model is not None:
        ax.loglog(s, model, color=SIM_COLOR, lw=2, label="Simulation")
    ax.set(xlabel="Genomic distance s (kb)", ylabel="Contact frequency P(s), normalised", title="Contact decay")
    ax.legend()


def _ins(ax, geom, w, data, model=None, b_data=(), b_model=()):
    """Insulation tracks with boundary ticks (boundaries are bead positions)."""
    x = lambda i: geom.axis_mb()[np.asarray(i, int)]
    pos = geom.axis_mb()[w:len(geom.axis_mb()) - w]
    ax.plot(pos, data, color=EXP_COLOR, lw=1.6, label="Hi-C")
    if model is not None:
        ax.plot(pos, model, color=SIM_COLOR, lw=1.6, label="Simulation")
    top = max(np.nanmax(data), np.nanmax(model) if model is not None else -9)
    for b, c, dy in ((b_data, EXP_COLOR, 0.0), (b_model, SIM_COLOR, 0.08)):
        b = np.asarray([i for i in b if 0 <= i < len(pos) + 2 * w], int)
        ax.plot(x(b), np.full(len(b), top + 0.1 + dy), "v", color=c, ms=5, ls="none")
    ax.set(xlabel=f"Position on {geom.chrom} (Mb)", ylabel="Insulation (log2)", title="Insulation and TAD boundaries")
    ax.legend()


def hic_input_figure(H, geom, info):
    """The Hi-C map read for this region: map, decay, insulation."""
    from analysis.maps import fill_masked, insulation_score, ps_curve
    h = fill_masked(H)
    fig, ax = make_fig(1, 3, cell=(5.0, 4.4))
    heatmap(fig, ax[0], _norm(h), geom, f"Hi-C ({info['resolution']:,} bp, {info['norm']})", "Contacts / 99th percentile", gamma=0.4)
    ps = ps_curve(h)[1:len(h) // 2]
    _decay(ax[1], geom, ps / ps.sum())
    w = 5
    ins = insulation_score(h, w)[w:-w]
    _ins(ax[2], geom, w, np.log2(np.clip(ins, 1e-12, None) / ins.mean()))
    save_figure(fig, "hic_input")


def hic_validation_figure(r, geom, source):
    """Simulation (lower triangle) against Hi-C (upper triangle), the decay and the insulation of both."""
    fig, ax = make_fig(1, 3, cell=(5.0, 4.4))
    comb = np.triu(_norm(r["h"]), 1) + np.tril(_norm(r["m"]), -1)
    heatmap(fig, ax[0], comb, geom, f"Hi-C (upper) vs {source} (lower)", "Contacts / 99th percentile", gamma=0.4)
    _decay(ax[1], geom, r["ps_data"], r["ps_model"])
    _ins(ax[2], geom, r["window"], r["ins_data"], r["ins_model"], r["b_data"], r["b_model"])
    save_figure(fig, "hic_validation")


def hic_correlation_figure(r, geom, source):
    """Correlation matrices of the O/E maps: Hi-C (upper) against the simulation (lower), and how well each bead's contact pattern agrees."""
    fig, ax = make_fig(1, 2, cell=(6.0, 4.6), width_ratios=[1, 1.15])
    comb = np.triu(r["corr_h"], 1) + np.tril(r["corr_m"], -1)
    heatmap(fig, ax[0], comb, geom, f"Hi-C (upper) vs {source} (lower)", "Correlation of contact patterns", diverging=True, vmax=0.6)
    x = geom.axis_mb()
    ax[1].fill_between(x, r["row_r"], 0, color=VIOLET, alpha=0.25, lw=0)
    ax[1].plot(x, r["row_r"], color=VIOLET, lw=1.8)
    ax[1].axhline(0, color=MUTED, lw=0.9)
    ax[1].axhline(np.nanmean(r["row_r"]), color=EXP_COLOR, lw=1.2, ls="--", label=f"mean r = {np.nanmean(r['row_r']):.2f}")
    ax[1].set(title="Where the contact patterns agree", ylabel="r between the two rows of a bead", xlabel=f"Position on {geom.chrom} (Mb)", xlim=(x[0], x[-1]), ylim=(-1, 1))
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    save_figure(fig, "hic_correlation")


def hic_scores_figure(results):
    """Bars of the agreement scores for every simulated map (higher is better)."""
    fig, ax = make_fig(1, 1, cell=(8.0, 4.2))
    a, names = ax[0], list(results)
    h = 0.8 / len(names)
    cols = [VIOLET, SIM_COLOR, MUTED]
    for k, name in enumerate(names):
        v = np.nan_to_num([results[name][key] for _, key, _ in METRICS])
        a.barh(np.arange(len(METRICS)) + k * h, v, h * 0.9, color=cols[k % 3], label=name)
    a.set_yticks(np.arange(len(METRICS)) + h * (len(names) - 1) / 2, [m[0] for m in METRICS])
    a.invert_yaxis()
    a.axvline(0, color=MUTED, lw=1)
    a.set(xlim=(-1, 1), xlabel="Agreement with the Hi-C map (1 = identical, 0 = unrelated, negative = opposite)", title="Agreement with Hi-C")
    a.legend()
    save_figure(fig, "hic_scores")
