"""Figures of the scan report (white background, purple theme). Every function returns PNG bytes; the plot type follows the design."""
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from loopsage.sensitivity import analysis as A
from loopsage.sensitivity import metrics as M
from loopsage.viz.theme import A_RED, B_BLUE, HEAT, INK, MUTED, SEQ, VIOLET, apply_theme

matplotlib.use("Agg")
LINES = LinearSegmentedColormap.from_list("scan_lines", ["#c4b2f5", "#6a3de8", "#2a1470", "#0d0626"])
CAT = ["#6a3de8", "#1a1a1f", "#e0457b", "#1baf7a", "#f58a3c", "#2b62d9", "#a98df0", "#9b2fb0"]


def png(fig) -> bytes:
    """Figure -> PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _fig(w=7.2, h=4.2, **kw):
    apply_theme()
    return plt.subplots(figsize=(w, h), **kw)


def num(v) -> str:
    """Compact number for cell labels."""
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2g}"


def fmt(v) -> str:
    return v if isinstance(v, str) else f"{v:.3g}"


def _ax(res, name):
    return next(a for a in res.design.axes if a.label == name or a.name == name)


def _logx(df, ax):
    v = pd.to_numeric(df[ax.label], errors="coerce")
    return ax.numeric and (ax.log or (v.min() > 0 and v.max() / max(v.min(), 1e-12) >= 30))


def agg(df, metric, by):
    """Mean and standard error of a metric per group; ratios (linked pairs) are pooled over the counts."""
    by = [by] if isinstance(by, str) else by
    if metric in M.POOLED and M.POOLED[metric][0] in df.columns:
        num, den = M.POOLED[metric]
        g = df.groupby(by)[[num, den]].sum()
        p = (g[num] / g[den].where(g[den] > 0))
        return pd.DataFrame({"mean": p, "se": np.sqrt(p * (1 - p) / g[den].where(g[den] > 0)).fillna(0), "n": g[den]})
    g = df.groupby(by)[metric]
    return pd.DataFrame({"mean": g.mean(), "se": g.std(ddof=1).fillna(0) / np.sqrt(g.count()), "n": g.count()})


def _colors(n):
    return [LINES(x) for x in (np.linspace(0.12, 0.95, n) if n > 1 else [0.5])]


def traces(res, metric, axis):
    """The metric over the Monte Carlo steps, one line per value of the axis (others pooled), burn-in shaded."""
    ax_ = _ax(res, axis)
    ser = A.series_by(res, metric, ax_.name)
    fig, ax = _fig()
    cols = CAT if not ax_.numeric else _colors(len(ser))
    for (v, (st, mean, lo, hi)), c in zip(ser.items(), cols):
        ax.plot(st / 1e3, mean, color=c, lw=1.6, label=fmt(v))
        ax.fill_between(st / 1e3, lo, hi, color=c, alpha=0.12, lw=0)
    first = next(iter(res.records.values()))
    b = next(r["steps"][r["burnin_idx"]] for r in res.records.values() if r["ok"])
    ax.axvspan(0, b / 1e3, color="#f1eefb", zorder=0)
    ax.text(b / 2e3, ax.get_ylim()[1], "burn-in", ha="center", va="top", color=MUTED, fontsize=8)
    ax.set_xlabel("Monte Carlo step (thousands)"); ax.set_ylabel(A.label(metric)); ax.set_xlim(0, None)
    ax.legend(title=ax_.label, ncol=2 if len(ser) > 6 else 1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return png(fig)


def response(res, metric, axis, df=None):
    """The production mean of the metric against one axis, with 95% intervals; faint dots are single runs."""
    df, ax_ = res.table if df is None else df, _ax(res, axis)
    g = agg(df, metric, ax_.label)
    fig, ax = _fig(6.4, 3.9)
    if ax_.numeric:
        x = g.index.values.astype(float)
        jit = df[ax_.label].astype(float)
        ax.scatter(jit, df[metric], s=9, color=VIOLET, alpha=0.18, lw=0, zorder=2)
        ax.fill_between(x, g["mean"] - 1.96 * g["se"], g["mean"] + 1.96 * g["se"], color=VIOLET, alpha=0.18, lw=0)
        ax.plot(x, g["mean"], color=INK, lw=2, marker="o", ms=5, zorder=3)
        if _logx(df, ax_):
            ax.set_xscale("log")
    else:
        ax.bar(range(len(g)), g["mean"], yerr=1.96 * g["se"], color=VIOLET, alpha=0.85, capsize=4)
        ax.set_xticks(range(len(g)), [fmt(v) for v in g.index])
    ax.set_xlabel(ax_.label); ax.set_ylabel(A.label(metric))
    return png(fig)


def heatmap(res, metric, a, b, df=None):
    """The mean metric on a grid of two axes (other axes and replicates averaged)."""
    df, A1, B1 = res.table if df is None else df, _ax(res, a), _ax(res, b)
    g = agg(df, metric, [B1.label, A1.label])["mean"].unstack()
    fig, ax = _fig(5.6 + 0.12 * g.shape[1], 4.4)
    im = ax.imshow(g.values, origin="lower", aspect="auto", cmap=SEQ, interpolation="nearest")
    ax.set_xticks(range(g.shape[1]), [fmt(v) for v in g.columns], rotation=45 if g.shape[1] > 6 else 0)
    ax.set_yticks(range(g.shape[0]), [fmt(v) for v in g.index])
    ax.grid(False)
    ax.set_xlabel(A1.label); ax.set_ylabel(B1.label)
    if g.size <= 64:
        lo, hi = np.nanmin(g.values), np.nanmax(g.values)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, num(v), ha="center", va="center", fontsize=8, color="white" if (v - lo) / (hi - lo + 1e-12) > 0.5 else INK)
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label(A.label(metric)); cb.outline.set_visible(False)
    return png(fig)


def lines_by(res, metric, a, b, df=None):
    """The metric against axis a, one curve per value of axis b."""
    df, A1, B1 = res.table if df is None else df, _ax(res, a), _ax(res, b)
    g = agg(df, metric, [B1.label, A1.label])
    fig, ax = _fig(6.8, 4.0)
    keys = list(dict.fromkeys(g.index.get_level_values(0)))
    for k, c in zip(keys, CAT if not B1.numeric else _colors(len(keys))):
        d = g.loc[k]
        x = d.index.values.astype(float) if A1.numeric else np.arange(len(d))
        ax.errorbar(x, d["mean"], yerr=1.96 * d["se"], color=c, marker="o", ms=4, lw=1.7, capsize=2, label=fmt(k))
    if A1.numeric and _logx(df, A1):
        ax.set_xscale("log")
    if not A1.numeric:
        ax.set_xticks(range(len(d)), [fmt(v) for v in d.index])
    ax.set_xlabel(A1.label); ax.set_ylabel(A.label(metric))
    ax.legend(title=B1.label, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=2 if len(keys) > 8 else 1)
    return png(fig)


MAX_PANELS = 3        # a report figure never has more than three sub-figures


def main_effects(res, metric, oat=False):
    """One small panel per input (other inputs averaged out), at most three per figure; returns a list of figures."""
    axes = res.design.axes
    return [_main_effects(res, metric, axes[i:i + MAX_PANELS], oat) for i in range(0, len(axes), MAX_PANELS)]


def _main_effects(res, metric, axes, oat):
    """The mean metric in bins of each given input, 95% intervals."""
    df = res.table
    pts = res.design.points
    n = len(axes)
    fig, axs = plt.subplots(1, n, figsize=(2.7 * n + 0.8, 3.3), sharey=True, squeeze=False)
    apply_theme()
    ymin = []
    for k, (ax, a) in enumerate(zip(axs[0], axes)):
        d = df
        if oat:
            keep = {p["id"] for p in pts if p["group"] in (a.name, "baseline")}
            d = df[df["point"].isin(keep)]
        x = d[a.label]
        key = A.groups(x) if (a.numeric and x.nunique() > 12) else x
        g = d.assign(_g=key.values).groupby("_g")
        mean, se = g[metric].mean(), g[metric].std(ddof=1).fillna(0) / np.sqrt(g[metric].count())
        center = g[a.label].mean() if a.numeric else pd.Series(range(len(mean)), index=mean.index)
        order = np.argsort(center.values)
        xs, ms, ss = center.values[order], mean.values[order], se.values[order]
        ax.fill_between(xs, ms - 1.96 * ss, ms + 1.96 * ss, color=VIOLET, alpha=0.18, lw=0)
        ax.plot(xs, ms, color=VIOLET, lw=2, marker="o", ms=4)
        if a.numeric and _logx(d, a):
            ax.set_xscale("log")
        if not a.numeric:
            ax.set_xticks(xs, [fmt(v) for v in mean.index[order]], rotation=30)
        ax.set_xlabel(a.label)
        if k == 0:
            ax.set_ylabel(A.label(metric))
    fig.tight_layout()
    return png(fig)


def importance_map(imp, col="eta2", title="Share of variance"):
    """Heat map: inputs (rows) against metrics (columns)."""
    P = imp.pivot(index="axis", columns="metric", values=col)
    P = P[[m for m in list(M.METRICS) + list(M.MD_METRICS) + list(M.TOPO_METRICS) if m in P.columns]]
    P = P.loc[sorted(P.index, key=lambda a: -P.loc[a].sum())]
    fig, ax = _fig(1.15 * P.shape[1] + 2.6, 0.62 * P.shape[0] + 2.2)
    im = ax.imshow(P.values, cmap=SEQ, vmin=0, vmax=max(float(np.nanmax(P.values)), 0.3), aspect="auto")
    ax.set_xticks(range(P.shape[1]), [A.label(m).split(" (")[0] for m in P.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(P.shape[0]), P.index)
    ax.grid(False)
    ax.xaxis.tick_top() if False else None
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if v > 0.45 * max(np.nanmax(P.values), 0.3) else INK)
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label(title); cb.outline.set_visible(False)
    return png(fig)


def sobol_bars(sob, metrics):
    """Sobol indices (S1 alone, ST with interactions) per input; up to six metrics, three panels per figure."""
    metrics = metrics[:6]
    return [_sobol_bars(sob, metrics[i:i + MAX_PANELS]) for i in range(0, len(metrics), MAX_PANELS)]


def _sobol_bars(sob, metrics):
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(4.1 * n, 2.9), squeeze=False)
    apply_theme()
    for ax, m in zip(axs.ravel(), metrics):
        d = sob[sob.metric == m]
        y = np.arange(len(d))
        ax.barh(y + 0.2, d.ST.clip(0, 1.05), 0.38, xerr=[(d.ST - d.ST_lo).clip(0), (d.ST_hi - d.ST).clip(0)], color="#1a1a1f", label="total (ST)", error_kw=dict(lw=1))
        ax.barh(y - 0.2, d.S1.clip(0, 1.05), 0.38, xerr=[(d.S1 - d.S1_lo).clip(0), (d.S1_hi - d.S1).clip(0)], color=VIOLET, label="alone (S1)", error_kw=dict(lw=1))
        ax.set_yticks(y, d.axis); ax.set_xlim(0, 1.1); ax.set_title(A.label(m).split(" (")[0], fontsize=10)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs[0, 0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return png(fig)


def tradeoff(res, mx, my, color_axis):
    """Metric against metric over the points, coloured by an input: how outputs move together."""
    pts = A.per_point(res)
    ax_ = _ax(res, color_axis)
    fig, ax = _fig(5.8, 4.2)
    c = pd.to_numeric(pts[ax_.label], errors="coerce")
    if ax_.numeric:
        norm = None if not _logx(pts, ax_) else matplotlib.colors.LogNorm()
        sc = ax.scatter(pts[mx], pts[my], c=c, cmap=LINES, s=34, lw=0, alpha=0.9, norm=norm)
        cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label(ax_.label); cb.outline.set_visible(False)
    else:
        for v, col in zip(sorted(pts[ax_.label].unique()), CAT):
            d = pts[pts[ax_.label] == v]
            ax.scatter(d[mx], d[my], color=col, s=34, lw=0, label=fmt(v))
        ax.legend(title=ax_.label)
    r = pts[[mx, my]].corr(method="spearman").iloc[0, 1]
    ax.text(0.03, 0.96, f"Spearman {r:+.2f}", transform=ax.transAxes, va="top", fontsize=9, color=MUTED)
    ax.set_xlabel(A.label(mx)); ax.set_ylabel(A.label(my))
    return png(fig)


def profiles(res, axis, field, sites=None):
    """Per-bead profile (leg density or loop coverage) for each value of one axis, stacked; CTCF sites marked."""
    prof = A.profile_by(res, axis, field)
    ax_ = _ax(res, axis)
    n = len(next(iter(prof.values())))
    fig, ax = _fig(7.4, 0.55 * min(len(prof), 9) + 2.2)
    vals = list(prof)[:: max(1, len(prof) // 9)]
    top = max(float(np.max(prof[v])) for v in vals) or 1
    cols = CAT if not ax_.numeric else _colors(len(vals))
    for i, (v, c) in enumerate(zip(vals, cols)):
        y0 = i * 1.05
        ax.fill_between(range(n), y0, y0 + prof[v] / top, color=c, alpha=0.8, lw=0)
        ax.text(-n * 0.01, y0 + 0.2, fmt(v), ha="right", va="center", fontsize=8, color=INK)
    if sites is not None:
        for b, o in sites:
            ax.axvline(b, color=A_RED if o == "forward" else B_BLUE, lw=0.8, alpha=0.5, ls=":")
    ax.set_yticks([]); ax.set_xlim(0, n); ax.set_xlabel("bead along the polymer"); ax.grid(False)
    ax.set_title(f"{axis} =", fontsize=9, loc="left", color=MUTED)
    return png(fig)


def loop_sizes(res, axis):
    """Distribution of loop lengths (log bins) for each value of one axis."""
    ax_ = _ax(res, axis)
    first = next(r for r in res.records.values() if r["ok"])
    edges = first["profiles"]["hist_edges"]
    prof = A.profile_by(res, axis, "loop_hist")
    fig, ax = _fig(6.4, 3.8)
    bp = first["bp"] / 1e3
    centre = np.sqrt(edges[:-1] * edges[1:]) * bp
    for (v, h), c in zip(prof.items(), CAT if not ax_.numeric else _colors(len(prof))):
        ax.plot(centre[:len(h)], h, color=c, lw=1.7, marker="o", ms=3, label=fmt(v))
    ax.set_xscale("log"); ax.set_xlabel("loop length (kb)"); ax.set_ylabel("share of loops")
    ax.legend(title=axis, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=2 if len(prof) > 8 else 1)
    return png(fig)


def convergence(res):
    """Three checks per run: effective samples, drift between the two halves, and the spread between replicates."""
    c = A.convergence(res)
    df = res.table
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.3))
    apply_theme()
    axs[0].hist(c.ess, bins=24, color=VIOLET, alpha=0.9); axs[0].axvline(30, color=A_RED, ls="--", lw=1)
    axs[0].set_xlabel("effective samples per run"); axs[0].set_ylabel("runs")
    axs[1].hist(c.drift.clip(upper=0.5) * 100, bins=24, color=VIOLET, alpha=0.9); axs[1].axvline(10, color=A_RED, ls="--", lw=1)
    axs[1].set_xlabel("drift between halves (% of the mean)")
    g = df.groupby("point")["mean_loop_length"]
    cv = (g.std(ddof=1) / g.mean().abs().clip(lower=1e-9)).dropna() * 100
    axs[2].hist(cv.clip(upper=100), bins=24, color=VIOLET, alpha=0.9); axs[2].set_xlabel("replicate spread (% of the mean)")
    fig.tight_layout()
    return png(fig)


def system_sketch(spec):
    """The synthetic polymer of the scan: beads, CTCF sites with their orientation, compartments."""
    from loopsage.sensitivity.runner import make_system
    box, _ = make_system(spec.system)
    n, bp = box.n_beads, box.bp_per_bead
    s = box.ctcf_sites
    fig, ax = _fig(8.0, 1.9 if box.compartments is None else 2.3)
    ax.plot([0, n], [0, 0], color=INK, lw=2, solid_capstyle="round")
    for pos, w, p in s.itertuples(index=False):
        x, d = pos / bp, 1 if p < 0.5 else -1
        c = A_RED if d == 1 else B_BLUE
        ax.plot([x, x], [0, 0.5], color=c, lw=2); ax.annotate("", xy=(x + 0.022 * n * d, 0.5), xytext=(x, 0.5), arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=11))
    if box.compartments is not None:
        comp = box.compartments
        ax.imshow(comp[None, :], extent=[0, n, -0.65, -0.3], cmap="coolwarm_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_ylim(-0.8 if box.compartments is not None else -0.3, 0.9); ax.set_xlim(0, n); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel(f"bead ({n} beads of {bp / 1e3:g} kb)  -  red arrow: forward CTCF, blue arrow: reverse CTCF")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return png(fig)


def leak_vs_barrier(res):
    """Crossing leaks against the excluded-volume barrier."""
    df = res.table
    fig, ax = _fig(6.0, 3.9)
    g = agg(df, "leak_rate", "ev_barrier")
    ax.scatter(df["ev_barrier"], df["leak_rate"], s=12, color=VIOLET, alpha=0.25, lw=0)
    ax.errorbar(g.index.values, g["mean"], yerr=1.96 * g["se"], color=INK, marker="o", ms=4, lw=1.6, capsize=2)
    ax.set_xlabel("strand-crossing barrier (kT)"); ax.set_ylabel("crossing leaks per 1000 MD steps")
    return png(fig)


# ---- phase transitions and bifurcations ----
def _mark(ax, row, axis_logx):
    """Shade the transition interval and mark its centre."""
    ax.axvspan(row.lo, row.hi, color="#f1eefb", zorder=0)
    ax.axvline(row.x_star, color=A_RED, lw=1.2, ls="--", zorder=1)


def phase_panel(res, row, d, axis):
    """Order parameter, its fluctuations and its autocorrelation time along the control parameter (three aligned panels)."""
    m, lg = row.metric, _logx(d, axis)
    g = d.groupby(axis.label)
    x = np.array(sorted(g.groups), float)
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 7.0), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1, 1], hspace=0.12))
    apply_theme()
    for ax, (col, lab) in zip(axs, ((m, A.label(m)), (f"std:{m}", "fluctuation (std over time)"), (f"tau:{m}", "autocorrelation time (steps)"))):
        ax.scatter(d[axis.label].astype(float), d[col], s=11, color=VIOLET, alpha=0.3, lw=0, zorder=2)
        mean = g[col].mean().reindex(x)
        ax.plot(x, mean, color=INK, lw=1.9, marker="o", ms=4, zorder=3)
        _mark(ax, row, lg)
        ax.set_ylabel(lab)
        if lg:
            ax.set_xscale("log")
    axs[0].set_title(f"{A.label(m)} along {axis.label}", loc="left", fontsize=11)
    axs[-1].set_xlabel(axis.label)
    return png(fig)


def bifurcation_diagram(res, row, d, axis, samples):
    """Snapshots of the order parameter at every control value; separate clouds at one value mean coexisting states."""
    m = row.metric
    fig, ax = _fig(6.6, 4.2)
    xs = sorted(samples)
    for xv in xs:
        v = samples[xv]
        ax.scatter(np.full(len(v), xv), v, s=5, color=VIOLET, alpha=0.12, lw=0, zorder=2)
    for r, (rep, g) in zip(range(99), d.groupby("rep")):
        ax.plot(g.groupby(axis.label)[m].mean().index, g.groupby(axis.label)[m].mean().values, color=CAT[r % len(CAT)], lw=1.3, marker="o", ms=3.5, label=f"replicate {rep}", zorder=3)
    _mark(ax, row, False)
    if _logx(d, axis):
        ax.set_xscale("log")
    ax.set_xlabel(axis.label); ax.set_ylabel(A.label(m))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="mean per replicate")
    return png(fig)


def distribution_at(res, row, samples, axis, k=6):
    """Distribution of the order parameter at k control values across the transition (offset histograms)."""
    xs = sorted(samples)
    sel = sorted({xs[int(i)] for i in np.linspace(0, len(xs) - 1, k)} | {x for x in xs if np.isclose(x, row.lo) or np.isclose(x, row.hi)})
    allv = np.concatenate([samples[x] for x in sel])
    bins = np.linspace(allv.min(), allv.max() + 1e-9, 28)
    fig, ax = _fig(6.0, 4.4)
    cols = _colors(len(sel))
    for i, (xv, c) in enumerate(zip(sel, cols)):
        h, _ = np.histogram(samples[xv], bins=bins, density=True)
        h = h / (h.max() + 1e-12) * 0.9
        ax.fill_between(0.5 * (bins[1:] + bins[:-1]), i, i + h, step="mid", color=c, alpha=0.85, lw=0)
        ax.plot(0.5 * (bins[1:] + bins[:-1]), i + h, color="white", lw=0.6, drawstyle="steps-mid")
    ax.set_yticks(range(len(sel)), [fmt(x) for x in sel]); ax.set_ylabel(axis.label)
    ax.set_xlabel(A.label(row.metric)); ax.grid(False)
    return png(fig)


def phase_map(res, metric, a, b, det):
    """Two axes: the mean (left) and the fluctuation (right) of the order parameter; dashed line = detected transition along `a`."""
    df, A1, B1 = res.table, _ax(res, a), _ax(res, b)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2))
    apply_theme()
    for ax, (col, title, cm) in zip(axs, ((metric, A.label(metric), SEQ), (f"std:{metric}", "fluctuation (std over time)", HEAT))):
        g = df.groupby([B1.label, A1.label])[col].mean().unstack()
        im = ax.imshow(g.values, origin="lower", aspect="auto", cmap=cm, interpolation="nearest")
        ax.set_xticks(range(g.shape[1]), [fmt(v) for v in g.columns], rotation=45 if g.shape[1] > 6 else 0)
        ax.set_yticks(range(g.shape[0]), [fmt(v) for v in g.index]); ax.grid(False)
        ax.set_xlabel(A1.label); ax.set_ylabel(B1.label); ax.set_title(title, loc="left", fontsize=10)
        cb = fig.colorbar(im, ax=ax, pad=0.02); cb.outline.set_visible(False)
        cols, rows = list(g.columns), list(g.index)
        for _, row in det.iterrows():                 # transition marker per row of b
            if row.bval in rows:
                ax.plot([np.interp(row.x_star, cols, range(len(cols)))], [rows.index(row.bval)], marker="D", ms=6, color=A_RED, mec="white", ls="")
    fig.tight_layout()
    return png(fig)


def binder_plot(res, metric, axis, size_axis):
    """Binder cumulant against the control value for several system sizes; their crossing marks a continuous transition."""
    df, fig_ax = res.table, _fig(6.0, 4.0)
    fig, ax = fig_ax
    sizes = sorted(df[size_axis.label].unique())
    for n, c in zip(sizes, _colors(len(sizes))):
        g = df[df[size_axis.label] == n].groupby(axis.label)[f"u4:{metric}"].mean()
        ax.plot(g.index, g.values, marker="o", ms=4, lw=1.6, color=c, label=fmt(n))
    ax.axhline(2 / 3, color=MUTED, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[0], 2 / 3, " ordered (2/3)", va="bottom", color=MUTED, fontsize=8)
    ax.set_xlabel(axis.label); ax.set_ylabel("Binder cumulant U4")
    if _logx(df, axis):
        ax.set_xscale("log")
    ax.legend(title=size_axis.label, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return png(fig)
