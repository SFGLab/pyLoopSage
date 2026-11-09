"""Phase transitions and bifurcations along a control parameter.

An *order parameter* (a metric) is followed along a numeric input. A transition is where it changes sharply; its signatures are
a peak of the fluctuations (susceptibility), a peak of the autocorrelation time (critical slowing down), and, for first-order-like
transitions, a bimodal distribution or replicates that fall onto different branches (bistability).
"""
import numpy as np
import pandas as pd

from loopsage.sensitivity import analysis as A
from loopsage.sensitivity import metrics as M

BIMODAL = 5 / 9          # Sarle's threshold
MIN_POINTS = 4
Z_ALL, Z_STEP = 5.0, 4.0     # significance (in standard errors) of the total change and of one step; strict, since many pairs are compared


def numeric_axes(res) -> list:
    """Axes along which a transition can be resolved: numeric with at least four distinct values."""
    out = []
    for a in res.design.axes:
        if a.numeric and res.table[a.label].nunique() >= MIN_POINTS:
            out.append(a)
    return out


def is_log(res, a) -> bool:
    v = res.table[a.label].astype(float)
    return a.log or (v.min() > 0 and v.max() / max(v.min(), 1e-12) >= 30)


def slices(res, axis):
    """[(label, dataframe)]: the runs along `axis` for each combination of the other inputs (grid); a single pooled slice otherwise."""
    df, others = res.table, [a.label for a in res.design.axes if a.label != axis.label]
    kind = res.design.kind
    if kind == "oat":
        keep = {p["id"] for p in res.design.points if p["group"] in (axis.name, "baseline")}
        return [("others at baseline", df[df["point"].isin(keep)])]
    if kind == "grid" and others and df[others].drop_duplicates().shape[0] <= 24:
        return [(", ".join(f"{o} = {v:.3g}" if not isinstance(v, str) else f"{o} = {v}" for o, v in zip(others, key if isinstance(key, tuple) else (key,))), g)
                for key, g in df.groupby(others)]
    if others:
        return [("pooled over the other inputs", df)]
    return [("", df)]


def _curve(d, metric, axis):
    g = d.groupby(axis.label)
    x = np.array(sorted(g.groups), float)
    mean = g[metric].mean().reindex(x).values
    se = (g[metric].std(ddof=1).fillna(0) / np.sqrt(g[metric].count())).reindex(x).values
    return x, mean, se


def _z(y, se, i, j):
    return abs(y[i] - y[j]) / max(np.hypot(se[i], se[j]), 1e-12)


def _runs(y, se, rng):
    """Merge neighbouring significant steps of the same sign into transitions: [(first, last)] index pairs."""
    step = [(abs(y[k + 1] - y[k]) >= 0.25 * rng and _z(y, se, k, k + 1) >= Z_STEP) * np.sign(y[k + 1] - y[k]) for k in range(len(y) - 1)]
    out, k = [], 0
    while k < len(step):
        if step[k]:
            e = k
            while e + 1 < len(step) and step[e + 1] == step[k]:
                e += 1
            out.append((k, e + 1))
            k = e + 1
        else:
            k += 1
    return out


def detect(res) -> pd.DataFrame:
    """One row per detected transition (metric, axis, slice): where it is, how sharp, and which signatures agree."""
    rows = []
    for axis in numeric_axes(res):
        logx = is_log(res, axis)
        for lab, d in slices(res, axis):
            if d[axis.label].nunique() < MIN_POINTS:
                continue
            for m in A.metric_cols(d):
                if d[m].notna().sum() < MIN_POINTS * 2:
                    continue
                x, y, se = _curve(d, m, axis)
                ok = np.isfinite(y)
                x, y, se = x[ok], y[ok], se[ok]
                if len(x) < MIN_POINTS:
                    continue
                xt = np.log10(x) if logx else x
                rng = float(np.ptp(y))
                oth = [a.label for a in res.design.axes if a.label != axis.label]
                bval = d[oth[0]].iloc[0] if len(oth) == 1 and d[oth[0]].nunique() == 1 else np.nan
                base = dict(metric=m, axis=axis.label, slice=lab, bval=bval)
                zmax = max(_z(y, se, i, j) for i in range(len(y)) for j in range(i + 1, len(y))) if rng > 1e-9 else 0
                runs = _runs(y, se, rng) if rng > 1e-9 and zmax >= Z_ALL else []
                if rng <= 1e-9 or zmax < Z_ALL:
                    rows.append(dict(base, kind="no significant change", direction="", x_star=np.nan, lo=np.nan, hi=np.nan, change=0.0, sharp=0.0,
                                     signatures="", bistable=False, strength=0.0, z=zmax))
                    continue
                if not runs:                                           # a significant but smooth change: report the steepest interval
                    k = int(np.argmax(np.abs(np.diff(y))))
                    runs = [(k, k + 1)]
                g = d.groupby(axis.label)
                chi, tau = g[f"std:{m}"].mean().reindex(x).values, g[f"tau:{m}"].mean().reindex(x).values
                bim = g[f"bim:{m}"].apply(lambda s: float((s > BIMODAL).mean())).reindex(x).values
                spread = g[m].agg(lambda s: (s.max() - s.min()) / rng if len(s) > 2 else 0.0).reindex(x).values
                for (i, j) in sorted(runs, key=lambda r: -abs(y[r[1]] - y[r[0]]))[:2]:
                    ks = int(np.argmax(np.abs(np.diff(y[i:j + 1])))) + i          # steepest interval inside the transition
                    sharp = float(abs(y[ks + 1] - y[ks]) / max(abs(y[j] - y[i]), 1e-12))
                    reg = slice(max(i - 1, 0), min(j + 2, len(y)))
                    chi_peak = bool(np.nanmax(chi[reg]) > 1.5 * np.nanmedian(chi) and np.nanmax(chi[reg]) >= np.nanmax(chi) * 0.999)
                    tau_peak = bool(np.nanmax(tau[reg]) > 1.5 * np.nanmedian(tau) and np.nanmax(tau[reg]) >= np.nanmax(tau) * 0.999)
                    near = [ks, ks + 1]
                    bistable = bool(spread[near].max() >= 0.4 and spread[near].max() >= 2 * np.median(spread))
                    bimod = bool(bim[reg].max() >= 0.5)
                    mid = (xt[ks] + xt[ks + 1]) / 2
                    kind = "sharp transition" if sharp >= 0.5 else "gradual crossover"
                    sigs = ", ".join(t for t, f in (("fluctuation peak", chi_peak), ("critical slowing down", tau_peak), ("replicates disagree (possible bistability)", bistable),
                                                    ("bimodal within runs", bimod)) if f)
                    step = abs(y[j] - y[i]) / rng
                    rows.append(dict(base, kind=kind, direction="rise" if y[j] > y[i] else "fall", x_star=float(10 ** mid if logx else mid), lo=float(x[i]), hi=float(x[j]),
                                     change=float((y[j] - y[i]) / (abs(y[i]) + 1e-12)), sharp=sharp, signatures=sigs, bistable=bistable, z=zmax,
                                     strength=min(zmax, 30) / 30 * step * sharp * (1 + chi_peak + tau_peak + bistable)))
    return pd.DataFrame(rows)


def classify(row) -> str:
    """Plain-language reading of one detected transition."""
    if row.kind == "no significant change":
        return "no significant change along this input"
    verb = "rises" if row.direction == "rise" else "falls"
    where = f"near {row.axis} = {row.x_star:.3g} ({row.lo:.3g} to {row.hi:.3g})"
    if row.kind == "gradual crossover":
        return f"{verb} gradually; steepest {where}. No sharp transition at this resolution."
    if row.bistable:
        return f"{verb} sharply {where}, and the replicates disagree there: possibly coexisting states (first-order transition or bifurcation); confirm with more replicates and a finer grid"
    if "fluctuation peak" in row.signatures and "critical slowing down" in row.signatures:
        return f"{verb} sharply {where}, with peaks of fluctuations and autocorrelation time: the signature of a continuous (critical-like) transition"
    if row.signatures:
        return f"{verb} sharply {where}; also seen: {row.signatures}"
    return f"{verb} sharply {where}; fluctuations do not peak there, so it may be a steep but smooth crossover. Refine the grid to tell."


def best(res, det, n=3):
    """The strongest candidates (distinct metric / axis / slice) to show in figures."""
    d = det[det.kind != "no significant change"].sort_values("strength", ascending=False)
    seen, out = set(), []
    for _, r in d.iterrows():
        if r.metric in seen:
            continue
        seen.add(r.metric)
        out.append(r)
        if len(out) >= n:
            break
    return out


def slice_df(res, row):
    """The runs of the slice a detected transition belongs to."""
    axis = next(a for a in res.design.axes if a.label == row.axis)
    for lab, d in slices(res, axis):
        if lab == row.slice:
            return axis, d
    return axis, res.table


def samples_by(res, d, metric, axis):
    """{x: array of snapshot samples} pooled over the runs in dataframe d (their stored subsamples)."""
    keys = {(int(p), int(r)) for p, r in zip(d["point"], d["rep"])}
    out = {}
    for t in res.tasks:
        if (t["point"], t["rep"]) in keys:
            rec = res.records[t["key"]]
            if rec["ok"] and metric in rec["samples"]:
                out.setdefault(t["values"][axis.name], []).append(rec["samples"][metric])
    return {k: np.concatenate(v) for k, v in sorted(out.items())}
