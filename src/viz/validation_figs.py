"""Validation figures: simulation against experiment (traces, correlation, binned trends) and a summary of all correlations."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde, rankdata, t as student_t

from common.utils import normalize_signal
from reporting.explanations import EXPLAIN
from validation.compare import smooth
from viz.theme import EXP_COLOR, INK as INK_LINE, make_fig, make_mosaic, MUTED, save_figure, SEQ, SIM_COLOR, VIOLET

TITLES = {"pearson": "Loop strength", "pearson_direct": "Per-loop strength", "ps_decay": "Contact decay P(s)", "insulation": "Insulation score", "insulation_anchor": "Boundary strength vs loops", "pc1": "Compartments"}
DENS = LinearSegmentedColormap.from_list("dens", SEQ(np.linspace(0.3, 1, 256)))          # density colours, never white


# ---- Building blocks -------------------------------------------------------

def scaled(v):
    """Signal rescaled to [0, 1] (lowest = 0, highest = 1): a readable axis, and the correlation does not change."""
    v = np.asarray(v, float)
    span = np.nanmax(v) - np.nanmin(v)
    return (v - np.nanmin(v)) / span if span > 0 else np.zeros_like(v)


def _fit(ax, x, y):
    """Least-squares line with its 95% confidence band."""
    n = len(x)
    if n < 4 or np.std(x) == 0:
        return
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 60)
    se = np.sqrt(((y - (a + b * x)) ** 2).sum() / (n - 2))
    band = student_t.ppf(0.975, n - 2) * se * np.sqrt(1 / n + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    ax.fill_between(xs, a + b * xs - band, a + b * xs + band, color=EXP_COLOR, alpha=0.12, lw=0)
    ax.plot(xs, a + b * xs, color=EXP_COLOR, lw=1.8)


def _dots(ax, x, y, color=None):
    """Scatter coloured by local density (or by `color`)."""
    if color is None:
        sel = np.random.default_rng(0).choice(len(x), min(len(x), 3000), replace=False)
        try:
            color = gaussian_kde(np.vstack([x[sel], y[sel]]))(np.vstack([x, y]))
        except np.linalg.LinAlgError:
            color = np.ones(len(x))
        order = np.argsort(color)
        x, y, color = x[order], y[order], color[order]
    return ax.scatter(x, y, c=color, cmap=DENS, s=16, edgecolor="white", linewidth=0.3)


def _stats_box(ax, p):
    """Pearson r (smoothed and raw) and Spearman rho in a corner box."""
    raw, sm = p["raw"], p["smooth"]
    lines = [f"Pearson r = {(sm or raw)['pearson']:.2f}" + (f"  (smoothed {p['sigma']:g})" if sm else ""),
             f"Spearman = {(sm or raw)['spearman']:.2f}"]
    if sm:
        lines.append(f"raw: r = {raw['pearson']:.2f}")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.3 * (hi - lo))                              # head-room so the box never covers points
    ax.text(0.04, 0.97, "\n".join(lines), va="top", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MUTED, alpha=0.9, lw=0.6))


def _scatter_panel(ax, p, color=None, xlabel="Experiment (scaled 0-1)", ylabel="Simulation (scaled 0-1)"):
    c = p["smooth"] or p["raw"]
    x, y = scaled(c["exp"]), scaled(c["sim"])
    _dots(ax, x, y, color)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=":")                  # perfect agreement
    _fit(ax, x, y)
    _stats_box(ax, p)
    ax.set(title="Correlation", xlabel=xlabel, ylabel=ylabel)


# ---- Per-position signals (bead), P(s) curves (sep) and per-loop values (loop) ----

def _bead_figure(p, geom):
    """Overlay of both signals, their difference along the region, and the correlation."""
    fig, ax = make_mosaic([["ov", "sc"], ["res", "sc"]], size=(12, 6.6), width_ratios=[2, 1.1], height_ratios=[1.5, 1])
    x = (geom.start + (np.arange(len(p["exp"])) + 0.5) * geom.bp_per_bead) / 1e6
    sig = p["sigma"] if p["smooth"] else 0
    ze, zs = scaled(normalize_signal(smooth(p["exp"], sig))), scaled(normalize_signal(smooth(p["sim"], sig)))
    ax["ov"].plot(x, scaled(normalize_signal(smooth(p["exp"], 0))), color=EXP_COLOR, lw=0.7, alpha=0.25)
    ax["ov"].plot(x, scaled(normalize_signal(smooth(p["sim"], 0))), color=SIM_COLOR, lw=0.7, alpha=0.3)
    ax["ov"].fill_between(x, ze, zs, color=SIM_COLOR, alpha=0.10, lw=0)
    ax["ov"].plot(x, ze, color=EXP_COLOR, lw=2.2, label="Experiment")
    ax["ov"].plot(x, zs, color=SIM_COLOR, lw=2.2, label="Simulation")
    if "b_exp" in p:                                                   # domain boundaries (insulation minima)
        lo = -0.08
        ax["ov"].plot(x[p["b_exp"]], np.full(len(p["b_exp"]), lo), "v", color=EXP_COLOR, ms=7, label=f"boundaries, experiment ({len(p['b_exp'])})")
        ax["ov"].plot(x[p["b_sim"]], np.full(len(p["b_sim"]), lo - 0.08), "v", color=SIM_COLOR, ms=7, label=f"boundaries, simulation ({len(p['b_sim'])})")
    ax["ov"].set(ylabel="Signal (scaled 0-1)", xlim=(x[0], x[-1]), title=f"{TITLES[p['key']]}: experiment vs simulation" + (f" (smoothed {sig:g} beads)" if sig else ""))
    ax["ov"].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax["ov"].tick_params(labelbottom=False)
    d = zs - ze
    ax["res"].fill_between(x, d, where=d >= 0, color=SIM_COLOR, alpha=0.75, lw=0, label="simulation higher")
    ax["res"].fill_between(x, d, where=d < 0, color="#e0457b", alpha=0.75, lw=0, label="experiment higher")
    ax["res"].axhline(0, color=MUTED, lw=0.8)
    ax["res"].set(ylabel="Sim - exp (scaled)", xlabel=f"Position on {geom.chrom} (Mb)", xlim=(x[0], x[-1]), title="Where they differ")
    ax["res"].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    _scatter_panel(ax["sc"], p)
    ax["sc"].text(0.96, 0.04, f"mean |difference| = {np.abs(d).mean():.2f}", transform=ax["sc"].transAxes, ha="right", va="bottom", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MUTED, alpha=0.9, lw=0.6))
    if "overlap" in p and np.isfinite(p["overlap"]):
        ax["sc"].text(0.96, 0.12, f"{p['overlap']:.0%} of experimental boundaries\nhave a simulated one within 2 beads", transform=ax["sc"].transAxes, fontsize=8,
                      va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MUTED, alpha=0.9, lw=0.6))
    return fig


def _sep_figure(p, geom):
    """Left: both decays (log-log, relative to the shortest separation). Right: their ratio, i.e. where the simulated decay is too flat or too steep."""
    fig, axs = make_fig(1, 2, cell=(5.6, 4.4))
    kb = geom.bp_per_bead / 1e3
    s = np.arange(len(p["exp"]))[1:] * kb
    curves = {}
    for v, col, lab in ((p["exp"], EXP_COLOR, "Experiment"), (p["sim"], SIM_COLOR, "Simulation")):
        v = v[1:] / (v[1:][v[1:] > 0][0] if (v[1:] > 0).any() else 1)
        ok = v > 0
        axs[0].plot(s[ok], v[ok], color=col, lw=0.8, alpha=0.4)
        sm = smooth(v, p["sigma"], log=True) if p["smooth"] else v
        curves[lab] = sm
        if p["smooth"]:
            axs[0].plot(s[ok], sm[ok], color=col, lw=2.2, label=lab)
        else:
            axs[0].plot([], [], color=col, label=lab)
    axs[0].set(xscale="log", yscale="log", title="Contact decay with separation", xlabel="Separation (kb)", ylabel="Mean contact (relative to the shortest)")
    axs[0].legend(frameon=False, fontsize=8)
    e, m = curves["Experiment"], curves["Simulation"]
    ok = (e > 0) & (m > 0)
    r = np.log2(m[ok] / e[ok])
    lim = max(2.0, 1.3 * np.percentile(np.abs(r), 90))                          # a few noisy points at the longest separations must not set the scale
    r = np.clip(r, -lim, lim)
    axs[1].axhline(0, color=MUTED, lw=0.9)
    axs[1].fill_between(s[ok], r, where=r >= 0, color=SIM_COLOR, alpha=0.5, lw=0, label="simulation higher")
    axs[1].fill_between(s[ok], r, where=r < 0, color="#e0457b", alpha=0.6, lw=0, label="experiment higher")
    axs[1].plot(s[ok], r, color=INK_LINE, lw=1.2)
    axs[1].legend(frameon=False, fontsize=8, loc="lower left")
    axs[1].set(ylim=(-lim, lim))
    axs[1].set(xscale="log", title="Where the decays differ", xlabel="Separation (kb)", ylabel="log2 (simulation / experiment)")
    _stats_box(axs[1], p)
    return fig


def _loop_figure(p, geom):
    """Per-loop strength as percentiles (0 = weakest, 100 = strongest loop): a readable axis that does not depend on the units."""
    fig, axs = make_fig(1, 2, cell=(5.6, 4.4))
    pct = lambda v: 100 * (rankdata(v) - 0.5) / len(v)
    ze, zs = pct(p["raw"]["exp"]), pct(p["raw"]["sim"])
    hb = axs[0].hexbin(ze, zs, gridsize=24, mincnt=1, cmap=DENS, linewidths=0.2, edgecolors="white", extent=(0, 100, 0, 100))
    axs[0].plot([0, 100], [0, 100], color=MUTED, lw=0.9, ls=":")                          # perfect agreement
    _fit(axs[0], ze, zs)
    _stats_box(axs[0], p)
    fig.colorbar(hb, ax=axs[0], shrink=0.8, label="Loops")
    axs[0].set(title="Every loop: experiment vs simulation", xlabel="Experiment loop strength (percentile)", ylabel="Simulation loop strength (percentile)")
    edges = np.linspace(0, 100, 11)
    b = np.clip(np.digitize(ze, edges[1:-1]), 0, 9)
    keep = [k for k in range(10) if (b == k).sum() > 2]
    mx = np.array([ze[b == k].mean() for k in keep])
    q = np.array([np.percentile(zs[b == k], [25, 50, 75]) for k in keep])
    axs[1].fill_between(mx, q[:, 0], q[:, 2], color=VIOLET, alpha=0.2, lw=0, label="middle 50%")
    axs[1].plot(mx, q[:, 1], "-o", color=VIOLET, lw=2.2, ms=5, label="median")
    axs[1].plot([0, 100], [50, 50], color=MUTED, lw=0.9, ls=":")                           # no relation
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    axs[1].set(title="Simulation by experimental strength", xlabel="Experiment loop strength (percentile, decile mean)", ylabel="Simulation loop strength (percentile)")
    return fig


def signal_figure(p, geom, suffix="", source=""):
    """One validation figure; layout depends on the kind of signal (per bead, per separation, per loop). `suffix` names the source (MC / MD)."""
    fig = {"bead": _bead_figure, "sep": _sep_figure, "loop": _loop_figure}[p["kind"]](p, geom)
    n = (p["smooth"] or p["raw"])["n"]
    note = f"; correlation after {p['sigma']:g}-bead smoothing" if p["smooth"] else ""
    text = EXPLAIN.get(f"validation_{p['key']}", {})
    save_figure(fig, f"validation_{p['key']}{suffix}", title=f"Validation: {TITLES[p['key']]}" + (f" ({source})" if source else ""),
                caption=f"Experimental versus simulated {p['signal']} (n = {n}{note})" + (f", simulation from {source}." if source else "."),
                explain=text.get("explain"), section="validation")


def validation_summary(by_source):
    """Pearson r per comparison for every simulated map (MC, MD): smoothed value (raw when there is none) with its 95% CI."""
    colors = dict(zip(by_source, (VIOLET, "#1baf7a")))
    groups = {}
    for src, rows in by_source.items():
        for r in rows:
            if np.isfinite(r["ci_lo"]):
                g = groups.setdefault(r["label"].replace(", smoothed", ""), {})
                if src not in g or "smoothed" in r["label"]:          # prefer the smoothed row
                    g[src] = r
    if not groups:
        return
    fig, axs = make_fig(1, 1, cell=(8.6, 0.8 * len(groups) + 2.0))
    ax = axs[0]
    for lo, hi in ((0, 0.3), (0.5, 0.7)):                                          # bands: weak and strong agreement
        ax.axvspan(lo, hi, color="#f3f0fa", lw=0)
    n = len(by_source)
    for k, (name, g) in enumerate(groups.items()):
        y = len(groups) - 1 - k
        for j, (src, r) in enumerate(g.items()):
            dy, col = (j - (n - 1) / 2) * 0.28, colors[src]
            ax.errorbar(r["pearson"], y - dy, xerr=[[max(r["pearson"] - r["ci_lo"], 0.0)], [max(r["ci_hi"] - r["pearson"], 0.0)]], fmt="o", color=col, ecolor=col, capsize=3, ms=8, lw=1.6)
            ax.text(min(r["ci_hi"], 0.97) + 0.03, y - dy, f"{r['pearson']:.2f}", va="center", fontsize=8, color=col)
    ax.axvline(0, color=MUTED, lw=0.8)
    first = lambda g: next(iter(g.values()))
    ax.set_yticks(range(len(groups)), [f"{n_}\n{first(g)['signal']}" for n_, g in reversed(list(groups.items()))])
    ax.set(title="Agreement with experiment", xlabel="Pearson r (95% CI, smoothed where available)", xlim=(-0.5, 1.15), ylim=(-0.6, len(groups) - 0.4))
    for src, col in colors.items():
        ax.plot([], [], "o", color=col, label=src)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save_figure(fig, "validation_summary")
