"""Quality metrics of a run: time series over the snapshots (Monte Carlo) and scalars from the 3D structures (optional).

Add your own with `@metric("name", "label", "unit")` on a function `f(ctx) -> array (snapshots,)`; `ctx` holds `Ms`, `Ns`
(legs, LEFs x snapshots), `ufs`, `Es`, `Fs`, `Bs`, `Ks`, `n_beads`, `bp` (bp per bead), `anchor` (bool per bead) and `nest` (nesting).
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from loopsage.analysis.nesting import nesting


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    unit: str
    fn: object
    default: bool
    help: str


METRICS = {}


def metric(name, label, unit="", default=True, help=""):
    """Register a metric function f(ctx) -> array (snapshots,)."""
    def deco(fn):
        METRICS[name] = Metric(name, label, unit, fn, default, help)
        return fn
    return deco


def _kind(ctx):
    """Per-loop class: parent, child or orphan (valid loops only)."""
    n = ctx.nest
    return (n.valid & (n.order == 0) & (n.n_desc > 0)), (n.valid & (n.order > 0)), (n.valid & (n.order == 0) & (n.n_desc == 0))


@metric("mean_loop_length", "Mean loop length", "kb", help="Average length of the loops: how far the rings extrude.")
def _mean_loop(ctx):
    return np.abs(ctx.Ns - ctx.Ms).mean(0) * ctx.bp / 1e3


@metric("unfolded_fraction", "Unfolded fraction", "", help="Share of the polymer covered by no loop.")
def _unfolded(ctx):
    T, n = ctx.Ms.shape[1], ctx.n_beads
    d, t = np.zeros((n + 1, T)), np.tile(np.arange(T), ctx.Ms.shape[0])
    np.add.at(d, (ctx.Ms.ravel(), t), 1)
    np.add.at(d, (ctx.Ns.ravel(), t), -1)
    return 1.0 - (np.cumsum(d, axis=0)[:n] > 0).mean(0)


@metric("n_parents", "Parent loops", "", help="Loops that hold other loops and sit inside none.")
def _parents(ctx):
    return _kind(ctx)[0].sum(0).astype(float)


@metric("n_children", "Child loops", "", help="Loops that sit inside another loop (any order).")
def _children(ctx):
    return _kind(ctx)[1].sum(0).astype(float)


@metric("n_orphans", "Isolated loops", "", help="Loops that are inside no loop and hold none.")
def _orphans(ctx):
    return _kind(ctx)[2].sum(0).astype(float)


@metric("nesting_depth", "Mean nesting depth", "", help="Average number of loops that contain a loop (0 = outermost).")
def _depth(ctx):
    n = ctx.nest
    return (n.order * n.valid).sum(0) / np.maximum(n.valid.sum(0), 1)


@metric("bound_fraction", "Bound fraction", "", help="Share of ring legs that sit on (or next to) a CTCF site.")
def _bound(ctx):
    return (ctx.anchor[ctx.Ms].astype(float) + ctx.anchor[ctx.Ns]).mean(0) / 2


@metric("anchored_loop_fraction", "Anchored loops", "", help="Share of loops with both legs on CTCF sites.")
def _anchored(ctx):
    return (ctx.anchor[ctx.Ms] & ctx.anchor[ctx.Ns]).mean(0).astype(float)


@metric("energy_total", "Total energy", "a.u.", help="Effective energy of the loop configuration.")
def _energy(ctx):
    return ctx.Es.astype(float)


@metric("n_loops", "Distinct loops", "", default=False, help="Loops that are not duplicates of another loop.")
def _nloops(ctx):
    return ctx.nest.valid.sum(0).astype(float)


@metric("longest_loop", "Longest loop", "kb", default=False, help="The longest loop at each snapshot.")
def _longest(ctx):
    return np.abs(ctx.Ns - ctx.Ms).max(0) * ctx.bp / 1e3


@metric("folding_energy", "Folding energy", "a.u.", default=False)
def _fold(ctx):
    return ctx.Fs.astype(float)


@metric("binding_energy", "Binding energy", "a.u.", default=False)
def _bind(ctx):
    return ctx.Bs.astype(float)


@metric("crossing_energy", "Crossing energy", "a.u.", default=False)
def _cross(ctx):
    return ctx.Ks.astype(float)


# ---- 3D metrics (one number per run, from the molecular dynamics frames) -----------------------------

MD_METRICS = {
    "rg": ("Radius of gyration", "nm", "Size of the 3D structure."),
    "rg_scaled": ("Compaction Rg / N^(1/3)", "nm", "Size relative to a space-filling chain: smaller = more compact."),
    "ps_exponent": ("Contact decay exponent", "", "Slope of contact probability against genomic distance, P(s) ~ s^-a: larger = faster decay."),
    "anchor_enrichment": ("Anchor contact enrichment", "x", "Contact strength between CTCF sites relative to the average pair at the same range."),
    "md_energy": ("Potential energy", "kJ/mol", "Potential energy of the 3D structures."),
}


# ---- topology metrics (linking of loops and chain crossings; 3D mode with `topology = true`) ----------------

TOPO_METRICS = {
    "linked_fraction": ("Linked loop pairs", "share", "Share of disjoint loop pairs that are linked (|Lk| >= 1), pooled over all frames and replicates."),
    "linked_pairs": ("Linked pairs per frame", "pairs", "Mean number of linked loop pairs in a 3D frame."),
    "frames_with_link": ("Frames with a link", "share", "Share of the 3D frames that contain at least one linked pair."),
    "mean_abs_lk": ("Mean |Lk| of linked pairs", "", "How strongly linked pairs wind around each other (1 = a simple link)."),
    "closest_approach": ("Closest loop approach", "bonds", "Median closest distance between two disjoint loops, in bond lengths."),
    "leak_rate": ("Crossing leaks", "per 1000 steps", "Strand crossings with the excluded volume switched on: the chain leaks through itself."),
    "passages": ("Strand passages per frame", "events", "Crossings through a bead whose excluded volume was switched off (topoisomerase)."),
    "links_created": ("Links created per frame", "pairs", "New links produced by topoisomerase passage attempts ('local' mode)."),
    "links_removed": ("Links removed per frame", "pairs", "Links undone by topoisomerase passage attempts ('local' mode)."),
    "ev_barrier": ("Strand-crossing barrier", "kT", "Energy barrier two bonds must overcome to pass through each other (from the excluded-volume settings)."),
}
POOLED = {"linked_fraction": ("link_num", "link_den")}      # ratios are pooled over runs, not averaged


def topology_metrics(md, mc, cfg) -> dict:
    """Linking numbers of the loops in every 3D frame, and the crossing counters of the molecular dynamics."""
    from loopsage.knots.crossing import barrier_kT
    from loopsage.knots.linking import frame_links
    nf, num, den, with_link, lks, near = len(md.frame_idx), 0, 0, 0, [], []
    for k in range(nf):
        i = md.frame_idx[k]
        rows, n_pairs, _, nr = frame_links(md.coords[k], mc.Ms[:, i], mc.Ns[:, i])
        num, den, with_link = num + len(rows), den + n_pairs, with_link + int(len(rows) > 0)
        lks += list(np.abs(rows[:, 4]))
        near += list(nr[:, 1])
    bond = float(np.linalg.norm(np.diff(md.coords[0], axis=0), axis=1).mean())
    topo = md.meta.get("topology") or {}
    tot = lambda key: float(sum(topo.get(key, [0])))
    steps = max(cfg.sim_step * len(topo.get("leak", [])), 1)
    return dict(link_num=float(num), link_den=float(den), linked_fraction=num / den if den else np.nan, linked_pairs=num / max(nf, 1),
                frames_with_link=with_link / max(nf, 1), mean_abs_lk=float(np.mean(lks)) if lks else np.nan,
                closest_approach=float(np.median(near) / bond) if near else np.nan, leak_rate=1000 * tot("leak") / steps,
                passages=tot("passage") / max(nf, 1), links_created=tot("created") / max(nf, 1), links_removed=tot("removed") / max(nf, 1),
                ev_barrier=barrier_kT(cfg.ev_ff_strength, cfg.ev_ff_power, cfg.sim_temp))


def context(chain, geom, anchor):
    """Everything the metric functions read."""
    ms, ns = np.minimum(chain.Ms, chain.Ns), np.maximum(chain.Ms, chain.Ns)
    return SimpleNamespace(Ms=ms, Ns=ns, ufs=chain.ufs, Es=chain.Es, Fs=chain.Fs, Bs=chain.Bs, Ks=chain.Ks, n_beads=geom.n_beads,
                           bp=geom.bp_per_bead, anchor=anchor, nest=nesting(ms, ns))


def select(names) -> list:
    """Metric names from the scan file: 'default', 'all' or a list."""
    out = []
    for n in names:
        out += [m for m, v in METRICS.items() if v.default] if n == "default" else list(METRICS) if n == "all" else [n]
    unknown = [n for n in out if n not in METRICS and n not in MD_METRICS and n not in TOPO_METRICS]
    if unknown:
        raise ValueError(f"unknown metric '{unknown[0]}' (known: {', '.join(list(METRICS) + list(MD_METRICS) + list(TOPO_METRICS))})")
    return list(dict.fromkeys(out))


def profiles(ctx, burn):
    """Per-bead and per-length summaries over the production snapshots (small arrays)."""
    n, ms, ns = ctx.n_beads, ctx.Ms[:, burn:], ctx.Ns[:, burn:]
    legs = np.bincount(np.concatenate([ms.ravel(), ns.ravel()]), minlength=n).astype(float)
    legs /= legs.sum() or 1.0
    cover = np.zeros(n + 1)
    for t in range(ms.shape[1]):                                              # share of snapshots in which a bead lies inside a loop
        d = np.zeros(n + 1)
        np.add.at(d, ms[:, t], 1); np.add.at(d, ns[:, t], -1)
        cover[:n] += np.cumsum(d)[:n] > 0
    edges = np.unique(np.round(np.geomspace(2, max(n, 4), 21)).astype(int))
    lens = np.abs(ns - ms).ravel()
    hist = np.histogram(lens[lens > 0], bins=edges)[0].astype(float)
    return dict(leg_density=legs, coverage=cover[:n] / max(ms.shape[1], 1), loop_hist=hist / max(hist.sum(), 1), hist_edges=edges.astype(float))


def bimodality(x) -> float:
    """Sarle's bimodality coefficient (above 5/9 suggests two peaks); NaN for constant series."""
    from scipy.stats import kurtosis, skew
    x = np.asarray(x, float)
    n = len(x)
    if n < 8 or x.std() < 1e-12:
        return np.nan
    return float((skew(x, bias=False) ** 2 + 1) / (kurtosis(x, bias=False) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def binder(x) -> float:
    """Binder cumulant U4 = 1 - <d^4> / (3 <d^2>^2) of the fluctuations d around the mean: 2/3 for an ordered, 0 for a Gaussian (disordered) state."""
    d = np.asarray(x, float) - np.mean(x)
    s2 = np.mean(d ** 2)
    return float(1 - np.mean(d ** 4) / (3 * s2 ** 2)) if s2 > 1e-18 else np.nan


def tau_int(x, c: float = 5.0) -> float:
    """Integrated autocorrelation time (snapshots) with Sokal's automatic window; 1 = independent samples."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < 8 or x.std() < 1e-12:
        return 1.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 2 * np.cumsum(acf) - 1
    ok = np.arange(n) >= c * tau
    m = int(np.argmax(ok)) if ok.any() else n - 1
    return float(max(tau[m], 1.0))


def md_metrics(md, anchor_beads) -> dict:
    """Scalars from an MDResult."""
    n = md.coords.shape[1]
    A = md.avg_map
    s = np.arange(2, max(n // 4, 6))
    P = np.array([np.mean(np.diagonal(A, k)) for k in s])
    ok = P > 0
    slope = -np.polyfit(np.log(s[ok]), np.log(P[ok]), 1)[0] if ok.sum() > 3 else np.nan
    idx = np.where(anchor_beads)[0]
    sub = A[np.ix_(idx, idx)] if len(idx) > 1 else np.zeros((1, 1))
    ii, jj = np.indices(sub.shape)
    far = np.abs(idx[ii] - idx[jj]) >= 3
    ref = np.mean([np.mean(np.diagonal(A, k)) for k in range(3, n // 2)])
    enrich = float(sub[far].mean() / ref) if far.any() and ref > 0 else np.nan
    return dict(rg=float(md.rg.mean()), rg_scaled=float(md.rg.mean() / n ** (1 / 3)), ps_exponent=float(slope),
                anchor_enrichment=enrich, md_energy=float(np.mean(md.epot)))
