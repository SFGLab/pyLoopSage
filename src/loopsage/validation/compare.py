"""Experimental signals (from loops/sites) and their simulated counterparts.

- Correlation statistics between an experimental and a simulated signal.
- Diagnostics using CTCF motif orientation stored in the input file.
- Validate a simulated contact map against the experimental input file.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import kendalltau, mannwhitneyu, pearsonr, spearmanr

from loopsage.analysis.maps import insulation_profile, insulation_window, loop_map_from_counts, map_features, ps_curve, ps_normalized
from loopsage.common.utils import normalize_signal, pearson_ci, strength_label
from loopsage.logger import get_logger
from loopsage.preproc.readers import numeric_col, read_region_table, valid_prob


# ---- Experimental signals (from loops/sites) and their simulated counterparts ----

def detect_format(path) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".bedpe", ".bed", ".narrowpeak"):
        return ext[1:]
    raise ValueError(f"unsupported format '{ext}'")


def _beads(df, c1, c2, geom):
    return np.asarray(geom.bead_of(((df[c1] + df[c2]) // 2).to_numpy()))


def bedpe_pairs(df, geom):
    """(x, y, weight): anchor beads and weight of every loop."""
    return _beads(df, 1, 2, geom), _beads(df, 4, 5, geom), numeric_col(df, 6, 1.0)


def site_signal(mat, df, geom):
    """Per-bead site score (exp) and the bead's total simulated contacts (sim); BED/narrowPeak."""
    x, w = _beads(df, 1, 2, geom), numeric_col(df, 4, 1.0)
    exp, sim = np.zeros(geom.n_beads), np.zeros(geom.n_beads)
    np.add.at(exp, x, w)
    np.add.at(sim, x, mat[x].sum(axis=1) - mat[x, x])
    return exp, sim


def exp_ps(df, geom) -> np.ndarray:
    """Experimental P(s): mean loop weight per anchor distance."""
    x, y, w = bedpe_pairs(df, geom)
    d = np.abs(y - x)
    total, count = np.zeros(geom.n_beads), np.zeros(geom.n_beads)
    np.add.at(total, d, w); np.add.at(count, d, 1)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def oe_values(df, mat, geom):
    """Per-loop observed/expected: weight and simulated contact divided by their P(s), each rescaled to mean 1 (same units on both sides)."""
    x, y, w = bedpe_pairs(df, geom)
    d = np.abs(y - x)
    e, s = w / (exp_ps(df, geom)[d] + 1e-9), mat[x, y] / (ps_curve(mat)[d] + 1e-9)
    return x, y, e / max(e.mean(), 1e-12), s / max(s.mean(), 1e-12)


def aggregate_to_beads(x, y, exp_vals, sim_vals, n):
    """Mean (not sum) of per-loop values on each anchor bead."""
    out = []
    for vals in (exp_vals, sim_vals):
        tot, cnt = np.zeros(n), np.zeros(n)
        for b in (x, y):
            np.add.at(tot, b, vals); np.add.at(cnt, b, 1)
        out.append(np.divide(tot, cnt, out=np.zeros(n), where=cnt > 0))
    return out


# ---- Correlation statistics between an experimental and a simulated signal ----

def permutation_p(exp, sim, n_perm: int = 500, seed: int = 0) -> float:
    """Empirical p-value of |Pearson r| by shuffling the simulated signal."""
    rng, r0 = np.random.default_rng(seed), pearsonr(exp, sim)[0]
    hits = sum(abs(pearsonr(exp, rng.permutation(sim))[0]) >= abs(r0) for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def correlate(exp, sim, mask_exp=None, mask_sim=None, drop_zero=True, n_perm=0) -> dict:
    """Pearson / Spearman / Kendall of two signals.

    Positions where BOTH raw signals are exactly zero are dropped (they agree
    trivially). `mask_*` are the raw (pre-normalisation) vectors used to find them.
    Returns None if fewer than 3 usable points remain.
    """
    exp, sim = np.asarray(exp, float), np.asarray(sim, float)
    n_total = len(exp)
    keep = np.ones(n_total, bool)
    if drop_zero:
        me = exp if mask_exp is None else np.asarray(mask_exp, float)
        ms = sim if mask_sim is None else np.asarray(mask_sim, float)
        keep = ~((me == 0) & (ms == 0))
        exp, sim = exp[keep], sim[keep]
    n = len(exp)
    if n < 3 or np.std(exp) == 0 or np.std(sim) == 0:
        return None
    r, p = pearsonr(sim, exp)
    lo, hi = pearson_ci(r, n)
    return dict(n=n, n_total=n_total, pearson=float(r), p=float(p), ci_lo=lo, ci_hi=hi, r2=float(r * r),
                spearman=float(spearmanr(sim, exp)[0]), kendall=float(kendalltau(sim, exp)[0]),
                perm_p=permutation_p(exp, sim, n_perm) if n_perm else None,
                strength=strength_label(r), exp=exp, sim=sim, idx=np.flatnonzero(keep))


# ---- Diagnostics using CTCF motif orientation stored in the input file -----

def _prob(df, col):
    """Reverse-orientation probabilities of a column, or None if missing/invalid."""
    if col >= df.shape[1]:
        return None
    p = numeric_col(df, col, -1.0)
    return p if valid_prob(p) else None


def canonical_consistency(mat, df, geom):
    """Do convergent (forward-left / reverse-right) loops show stronger simulated contacts?"""
    p1, p2 = _prob(df, 7), _prob(df, 8)
    if p1 is None or p2 is None:
        return None
    x, y, _ = bedpe_pairs(df, geom)
    vals, known = mat[x, y], (p1 >= 0) & (p2 >= 0)
    can = known & (p1 < 0.4) & (p2 > 0.6)
    non = known & ~can
    if can.sum() < 3 or non.sum() < 3:
        return None
    p = mannwhitneyu(vals[can], vals[non], alternative="greater").pvalue
    return dict(name="Convergent-orientation loops are stronger", n=f"{can.sum()} vs {non.sum()}",
                value=f"mean {vals[can].mean():.3g} vs {vals[non].mean():.3g}", p=float(p),
                expect="convergent > other")


def _site_probs(df, fmt, geom):
    """(bead, reverse-probability) of every site with orientation info."""
    beads, probs = [], []
    cols = {"bedpe": [(7, (1, 2)), (8, (4, 5))], "bed": [(6, (1, 2))], "narrowpeak": [(5, (1, 2))]}[fmt]
    for col, (c1, c2) in cols:
        p = _prob(df, col)
        if p is not None:
            beads.append(_beads(df, c1, c2, geom))
            probs.append(np.where(p >= 0, 1 - p, -1.0) if fmt == "narrowpeak" else p)
    return (np.concatenate(beads), np.concatenate(probs)) if beads else (np.array([]), np.array([]))


def asymmetry_consistency(mat, df, fmt, geom, window=5):
    """Correlate site orientation with the right-minus-left contact asymmetry (expected < 0)."""
    beads, probs = _site_probs(df, fmt, geom)
    ok = probs >= 0
    if ok.sum() < 5:
        return None
    n = mat.shape[0]
    asym = np.array([mat[b, b + 1:min(n, b + window + 1)].sum() - mat[b, max(0, b - window):b].sum()
                     for b in beads[ok]])
    if np.std(asym) == 0:
        return None
    r, p = pearsonr(probs[ok], asym)
    return dict(name="Orientation vs contact asymmetry", n=str(int(ok.sum())), value=f"r = {r:.3f}",
                p=float(p), expect="negative r")


# ---- Validate a simulated contact map against the experimental input file ----

log = get_logger(__name__)


@dataclass
class ValidationResult:
    rows: list = field(default_factory=list)         # one dict per correlation (see correlate)
    plots: list = field(default_factory=list)        # dicts: key, signal, kind (bead / sep / loop), raw and smoothed correlation, signals
    diagnostics: list = field(default_factory=list)  # orientation tests
    notes: list = field(default_factory=list)


def _add(res, label, signal, c):
    """Store one correlation row; returns False (and a note) when there is too little data."""
    if c is None:
        res.notes.append(f"{label} ({signal}): too little data.")
        return False
    res.rows.append(dict(label=label, signal=signal, **{k: v for k, v in c.items() if k not in ("exp", "sim", "idx")}))
    return True


def smooth(v, sigma, log=False):
    """Gaussian smoothing in beads (log=True: in log1p space, for steep curves); sigma <= 0 returns v."""
    v = np.asarray(v, float)
    if sigma <= 0:
        return v
    return np.expm1(gaussian_filter1d(np.log1p(np.clip(v, 0, None)), sigma, mode="nearest")) if log else gaussian_filter1d(v, sigma, mode="nearest")


def loop_matrix(df, geom) -> np.ndarray:
    """Symmetric bead x bead matrix of the input loop weights (straight from the data, nothing inferred)."""
    n = geom.n_beads
    x = np.asarray(geom.bead_of(((df[1] + df[2]) // 2).to_numpy()))
    y = np.asarray(geom.bead_of(((df[4] + df[5]) // 2).to_numpy()))
    w = numeric_col(df, 6, 1.0)
    m = np.zeros((n, n))
    np.add.at(m, (x, y), w)
    return m + m.T


def experiment_map(df, geom, n_lef=None) -> np.ndarray:
    """Hi-C-like map of the input loops, drawn exactly like the simulated one (loop_map) with the weights scaled to the same total (2 per LEF)."""
    c = loop_matrix(df, geom)
    k = n_lef or max(int((c > 0).sum() // 2), 1)
    return loop_map_from_counts(c * (2 * k / max(c.sum(), 1e-12)), geom.n_beads)


def boundaries(ins, sigma: float = 1.5, prominence: float = 0.5) -> np.ndarray:
    """Beads at local minima of the z-scored insulation score (domain boundaries)."""
    z = normalize_signal(smooth(ins, sigma))
    return find_peaks(-z, prominence=prominence)[0]


def boundary_overlap(a, b, tol: int = 2) -> float:
    """Share of boundaries in `a` with a boundary of `b` within `tol` beads (nan if `a` is empty)."""
    return float(np.mean([np.abs(b - i).min() <= tol for i in a])) if len(a) and len(b) else float("nan")


def experiment_features(path, geom):
    """Map features (insulation, P(s)) of the experimental loop map, or None when the input is not a BEDPE file."""
    if detect_format(path) != "bedpe":
        return None
    return map_features(loop_matrix(read_region_table(path, geom, paired=True), geom))


def _pair(exp, sim, logr):
    """The two signals as correlated: log10 of the positive part for decay curves (logr), else as they are."""
    e, s = np.asarray(exp, float), np.asarray(sim, float)
    if logr:
        ok = (e > 0) & (s > 0)
        return np.log10(e[ok]), np.log10(s[ok])
    return e, s


def _compare(res, label, signal, key, exp, sim, kind, sigma, perm=0, logr=False, **extra):
    """Correlate the signals smoothed with `sigma` beads (the primary row, which the figure shows) and, as a reference, raw.

    kind "loop" (per-loop values) and logr (decay curves, a function of separation) are never smoothed.
    """
    z = (lambda v: v) if logr else normalize_signal
    sm = None
    if sigma > 0 and kind not in ("loop", "sep"):
        es, ss = smooth(exp, sigma), smooth(sim, sigma)
        sm = correlate(z(es), z(ss), exp, sim, True, perm)
    ce, cs = _pair(exp, sim, logr)
    raw = correlate(z(ce), z(cs), ce, cs, True, perm)
    if not _add(res, label, signal, sm or raw):
        return
    if sm:
        _add(res, f"{label}, raw", signal, raw)
    res.plots.append(dict(key=key, signal=signal, kind=kind, raw=raw, smooth=sm, sigma=sigma if sm else 0.0, exp=np.asarray(exp, float),
                          sim=np.asarray(sim, float), **extra))


def validate(mat, cfg, geom, path=None, comp=None, track=None, track_name="PC1 of the simulated map") -> ValidationResult:
    """Correlate simulated vs experimental loop strength, P(s) decay, insulation and compartments (path overrides cfg.interaction_file).

    Per-position profiles are smoothed with one Gaussian width (cfg.track_smooth beads, as the Hi-C insulation and A/B tracks) before they are
    correlated and plotted; the raw correlation is kept as a reference row. P(s) curves are compared as normalised probabilities (sum 1).
    """
    path = path or cfg.interaction_file
    fmt = detect_format(path)
    df = read_region_table(path, geom, paired=(fmt == "bedpe"))
    res, sigma = ValidationResult(), float(cfg.track_smooth)
    paired = fmt == "bedpe"

    if paired:
        x, y, e_oe, s_oe = oe_values(df, mat, geom)
        exp, sim = aggregate_to_beads(x, y, e_oe, s_oe, geom.n_beads)
        name = "loop strength (O/E)"
    else:
        exp, sim = site_signal(mat, df, geom)
        name = "site strength"
    _compare(res, "Primary (zero-excluded)", name, "pearson", exp, sim, "bead", sigma, perm=500)
    _add(res, "Reference (zeros kept, raw)", name, correlate(normalize_signal(exp), normalize_signal(sim), drop_zero=False))

    if paired and len(x) > 1:
        c = correlate(normalize_signal(e_oe), normalize_signal(s_oe), e_oe, s_oe, True, 500)
        if _add(res, "Per-loop (no aggregation)", "loop strength (O/E)", c):
            res.plots.append(dict(key="pearson_direct", signal="loop strength (O/E)", kind="loop", raw=c, smooth=None, sigma=0.0, exp=e_oe, sim=s_oe))
        ps_e, ps_s = ps_normalized(experiment_map(df, geom, cfg.n_lef)), ps_normalized(mat)         # both from a map, both probabilities (sum 1)
        _compare(res, "P(s) decay", "contact vs bead separation", "ps_decay", np.r_[0, ps_e], np.r_[0, ps_s], "sep", sigma, logr=True)
        w = insulation_window(geom)                                   # same diamond (in bp) on both maps; the raw loop dots are not a map
        ins_sim, ins_exp = insulation_profile(mat, w), insulation_profile(experiment_map(df, geom, cfg.n_lef), w)
        be, bs = boundaries(smooth(ins_exp, sigma)), boundaries(smooth(ins_sim, sigma))             # boundaries of the smoothed scores shown in the figure
        _compare(res, "Insulation score", "insulation score, experiment (loop map) vs simulation", "insulation", ins_exp, ins_sim, "bead", sigma,
                 b_exp=be, b_sim=bs, overlap=boundary_overlap(be, bs), overlap_rev=boundary_overlap(bs, be))
        _compare(res, "Boundary strength vs loops", "loop strength vs simulated boundary strength (minus insulation)", "insulation_anchor", exp, -ins_sim, "bead", sigma)
        res.diagnostics.append(canonical_consistency(mat, df, geom))
    if comp is not None and track is not None and np.std(comp) > 0 and np.std(track) > 0:     # simulated compartment track vs the input A/B track
        _compare(res, "Compartments", f"{track_name} vs input compartment track", "pc1", comp, track, "bead", sigma)
    res.diagnostics.append(asymmetry_consistency(mat, df, fmt, geom))
    res.diagnostics = [d for d in res.diagnostics if d]
    log.info("Validation: %d correlations, %d diagnostics (smoothing %.1f beads)", len(res.rows), len(res.diagnostics), sigma)
    return res
