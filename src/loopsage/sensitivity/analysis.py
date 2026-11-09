"""Turn the runs of a scan into tables and sensitivity measures (first-order effects, Sobol indices, elasticity, convergence)."""
import numpy as np
import pandas as pd

from loopsage.sensitivity import metrics as M


def metric_cols(df) -> list:
    """Metric columns of the run table (Monte Carlo first, then 3D)."""
    return [c for c in list(M.METRICS) + list(M.MD_METRICS) + list(M.TOPO_METRICS) if c in df.columns]


def label(name) -> str:
    """Readable name and unit of a metric."""
    if name in M.METRICS:
        m = M.METRICS[name]
        return f"{m.label} ({m.unit})" if m.unit else m.label
    if name in M.TOPO_METRICS:
        lab, unit, _ = M.TOPO_METRICS[name]
        return f"{lab} ({unit})" if unit else lab
    if name in M.MD_METRICS:
        lab, unit, _ = M.MD_METRICS[name]
        return f"{lab} ({unit})" if unit else lab
    return name


def per_point(res) -> pd.DataFrame:
    """Mean and standard error over the replicates for every point."""
    df, axes = res.table, [a.label for a in res.design.axes]
    cols = metric_cols(df)
    g = df.groupby("point")
    out = g[axes].first()
    for c in cols:
        out[c] = g[c].mean()
        out[c + "_se"] = g[c].std(ddof=1).fillna(0) / np.sqrt(g[c].count())
    for name, (num, den) in M.POOLED.items():                       # ratios: sum of counts over replicates (not a mean of ratios)
        if name in cols and num in df.columns:
            n, d = g[num].sum(), g[den].sum()
            p = (n / d.where(d > 0)).astype(float)
            out[name], out[name + "_se"] = p, np.sqrt(p * (1 - p) / d.where(d > 0)).fillna(0)
            out[den] = d
    out["group"] = [res.design.points[p]["group"] for p in out.index]
    out["row"] = [res.design.points[p]["row"] for p in out.index]
    return out


def groups(x, bins: int = 8):
    """Group labels of a variable: its values when few, else quantile bins."""
    x = pd.Series(x)
    if x.dtype == object or x.nunique() <= 12:
        return x.astype(str)
    return pd.qcut(x, bins, duplicates="drop").astype(str)


def eta2(y, x) -> float:
    """Share of the variance of y explained by the group means of x (first-order effect, 0 to 1)."""
    y = np.asarray(y, float)
    tot = y.var()
    if tot < 1e-15:
        return 0.0
    g = pd.DataFrame({"y": y, "g": groups(x).values}).groupby("g")["y"]
    return float(np.clip((g.count() * (g.mean() - y.mean()) ** 2).sum() / (tot * len(y)), 0, 1))


def spearman(y, x):
    """Rank correlation (NaN for a categorical x)."""
    try:
        xs = pd.Series(x).astype(float)
    except (TypeError, ValueError):
        return np.nan
    return float(pd.Series(y).corr(xs, method="spearman")) if xs.nunique() > 1 else np.nan


def importance(res) -> pd.DataFrame:
    """Rows: metric x axis. eta2 (first-order share of variance), rho (Spearman), range (change of the mean across the axis, relative to the overall mean)."""
    df, rows = res.table, []
    for m in metric_cols(df):
        ok = df[m].notna()
        y = df.loc[ok, m].values
        if len(y) < 4:
            continue
        for a in res.design.axes:
            x = df.loc[ok, a.label]
            mean = df.groupby(x.astype(str))[m].mean()
            rng = (mean.max() - mean.min()) / (abs(y.mean()) + 1e-12)
            rows.append(dict(metric=m, axis=a.label, eta2=eta2(y, x.values), rho=spearman(y, x.values), rel_range=float(rng)))
    return pd.DataFrame(rows)


def sobol(res, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """First-order (S1) and total (ST) Sobol indices from a sobol design (Saltelli / Jansen estimators) with bootstrap intervals."""
    pts = per_point(res)
    axes, rng = res.design.axes, np.random.default_rng(seed)
    rows = []
    for m in metric_cols(pts):
        get = lambda role: pts[pts["group"] == role].sort_values("row")[m].values
        A, B = get("A"), get("B")
        var0 = np.concatenate([A, B]).var()
        for i, a in enumerate(axes):
            AB = get(f"AB{i}")
            n = min(len(A), len(B), len(AB))
            if var0 < 1e-15 or n < 8:
                rows.append(dict(metric=m, axis=a.label, S1=0.0, ST=0.0, S1_lo=0.0, S1_hi=0.0, ST_lo=0.0, ST_hi=0.0)); continue
            est = lambda idx: (np.mean(B[idx] * (AB[idx] - A[idx])) / np.concatenate([A[idx], B[idx]]).var(),
                               0.5 * np.mean((A[idx] - AB[idx]) ** 2) / np.concatenate([A[idx], B[idx]]).var())
            s1, st = est(np.arange(n))
            boots = np.array([est(rng.integers(0, n, n)) for _ in range(n_boot)])
            lo, hi = np.percentile(boots, [5, 95], axis=0)
            rows.append(dict(metric=m, axis=a.label, S1=s1, ST=st, S1_lo=lo[0], S1_hi=hi[0], ST_lo=lo[1], ST_hi=hi[1]))
    return pd.DataFrame(rows)


def elasticity(x, y) -> float:
    """Log-log slope: percent change of y per percent change of x (NaN unless both are positive and x varies)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]) if ok.sum() > 2 and np.ptp(np.log(x[ok])) > 1e-9 else np.nan


def insights(res, imp) -> list:
    """One plain sentence per metric: the input that matters most, in which direction, and how much."""
    out = []
    for m in metric_cols(res.table):
        d = imp[imp.metric == m].sort_values("eta2", ascending=False)
        top = d.iloc[0]
        if top.eta2 < 0.05:
            out.append((m, f"<b>{label(m).split(' (')[0]}</b> hardly changes across the scanned ranges (largest single effect: {top.axis}, {top.eta2:.0%} of the variance)."))
            continue
        rho = top.rho
        way = "" if np.isnan(rho) else (" It rises with it." if rho > 0.5 else " It falls as it grows." if rho < -0.5 else " The trend is not monotonic (it peaks or dips in the middle).")
        second = f" Next: <b>{d.iloc[1].axis}</b> ({d.iloc[1].eta2:.0%})." if len(d) > 1 and d.iloc[1].eta2 >= 0.05 else ""
        rest = max(1 - d.eta2.sum(), 0)
        inter = f" About {rest:.0%} is interaction between inputs or noise." if len(d) > 1 and rest > 0.15 else ""
        out.append((m, f"<b>{label(m).split(' (')[0]}</b> is governed mostly by <b>{top.axis}</b> ({top.eta2:.0%} of the variance; the mean changes by {top.rel_range:.0%} across the range).{way}{second}{inter}"))
    return out


def convergence(res) -> pd.DataFrame:
    """Per run: effective samples and the drift between the first and second half of the production run (mean loop length)."""
    rows = []
    for t in res.tasks:
        r = res.records.get(t["key"])
        if not r or not r["ok"] or "mean_loop_length" not in r["series"]:
            continue
        n = len(r["series"]["mean_loop_length"]) - r["burnin_idx"]
        h1, h2 = r["first_half"]["mean_loop_length"], r["second_half"]["mean_loop_length"]
        rows.append(dict(point=t["point"], rep=t["rep"], ess=n / r["tau"]["mean_loop_length"], tau=r["tau"]["mean_loop_length"],
                         drift=abs(h2 - h1) / (abs(r["summary"]["mean_loop_length"]) + 1e-9), n=n))
    return pd.DataFrame(rows)


def series_by(res, metric, axis, where=None):
    """Time series of a metric grouped by the values of one axis: {value: (steps, mean, lo, hi)}; replicates (and axes not in `where`) pooled.

    `where` = {axis name: value} keeps only the runs with those values.
    """
    a = next(x for x in res.design.axes if x.label == axis or x.name == axis)
    groups_ = {}
    for t in res.tasks:
        r = res.records.get(t["key"])
        if where and any(t["values"].get(k) != v for k, v in where.items()):
            continue
        if r and r["ok"] and metric in r["series"]:
            groups_.setdefault(t["values"][a.name], []).append((r["steps"], r["series"][metric]))
    out = {}
    for v, lst in groups_.items():
        L = min(len(s) for _, s in lst)
        arr = np.array([s[:L] for _, s in lst])
        out[v] = (lst[0][0][:L], arr.mean(0), np.percentile(arr, 15, axis=0), np.percentile(arr, 85, axis=0))
    return dict(sorted(out.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])))


def profile_by(res, axis, field):
    """Mean per-bead (or per-length) profile for each value of one axis: {value: array}; needs a constant number of beads."""
    a = next(x for x in res.design.axes if x.label == axis or x.name == axis)
    acc = {}
    for t in res.tasks:
        r = res.records.get(t["key"])
        if r and r["ok"]:
            acc.setdefault(t["values"][a.name], []).append(r["profiles"][field])
    out = {}
    for v, lst in acc.items():
        if len({len(x) for x in lst}) == 1:
            out[v] = np.mean(lst, axis=0)
    return dict(sorted(out.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])))
