"""Simulated contact map against a Hi-C map on the same bead grid (NaN = masked beads)."""
import numpy as np

from scipy.ndimage import gaussian_filter1d

from analysis.maps import compartment_from_hic, oe_correlation, STATE_SMOOTH, fill_masked, insulation_score, observed_over_expected, pc1, ps_curve
from validation.compare import boundaries, boundary_overlap

NAN = float("nan")
METRICS = [("log-contact r", "r_log", "Pearson r of log contacts over all bead pairs"),
           ("O/E r", "r_oe", "Pearson r after dividing each diagonal by its mean (decay removed)"),
           ("SCC", "scc", "mean Pearson r per diagonal (stratified correlation)"),
           ("P(s) r", "r_ps", "Pearson r of the log contact decay with distance"),
           ("insulation r", "r_ins", "Pearson r of the insulation scores (domain boundaries)"),
           ("boundary F1", "f1", "TAD boundaries found in both maps"),
           ("correlation matrix r", "r_cm", "Pearson r between the two O/E correlation matrices (which beads share a contact pattern: compartments and domains)"),
           ("compartment r", "r_pc", "Pearson r of the Hi-C leading eigenvector (A/B) with the average Potts state when the epigenetic model is on (B = negative), else with the simulated eigenvector (sign free)")]


def _r(x, y) -> float:
    """Pearson r, NaN-safe (NaN when degenerate)."""
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 3 and np.std(x[ok]) > 1e-12 and np.std(y[ok]) > 1e-12 else NAN


def _scc(h, m, kmin=2):
    """Mean per-diagonal Pearson r (diagonals weighted by their number of pairs)."""
    n, r, w = len(h), [], []
    for k in range(kmin, n // 2):
        x, y = np.diagonal(h, k), np.diagonal(m, k)
        v = _r(x, y)
        if np.isfinite(v):
            r.append(v)
            w.append(len(x))
    return float(np.average(r, weights=w)) if r else NAN


def compare_hic(model, hic, window: int = 5, tol: int = 2, state=None, track: float = 0.0) -> dict:
    """Metrics and the tracks behind them for a simulated map `model` and the Hi-C map `hic` (same shape).

    `track` = smoothing (beads) of the insulation score and of the eigenvector.
    `state` = average Potts state per bead: with the epigenetic model on, the Hi-C compartments are compared with it instead of with the simulated eigenvector."""
    n = len(hic)
    seen = np.isfinite(hic)
    h, m = fill_masked(hic), np.array(model, float)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    up = (sep >= 2) & (np.triu(np.ones((n, n)), 1) > 0) & seen
    ok = up & (h > 0) & (m > 0)
    oe_h, oe_m = observed_over_expected(h), observed_over_expected(m)
    ps_h, ps_m = ps_curve(h)[1:n // 2], ps_curve(m)[1:n // 2]
    ps_h, ps_m = ps_h / ps_h.sum(), ps_m / ps_m.sum()
    ins_h, ins_m = insulation_score(h, window, track), insulation_score(m, window, track)
    sl = slice(window, n - window)
    ins_h, ins_m = (np.nan_to_num(np.log2(np.clip(v[sl], 1e-12, None) / v[sl].mean())) for v in (ins_h, ins_m))
    b_h, b_m = boundaries(ins_h), boundaries(ins_m)
    rec, prec = boundary_overlap(b_h, b_m, tol), boundary_overlap(b_m, b_h, tol)
    f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
    c_h, c_m = oe_correlation(h), oe_correlation(m)
    iu = (sep >= 2) & (np.triu(np.ones((n, n)), 1) > 0)
    row_r = np.array([_r(np.delete(c_h[i], i), np.delete(c_m[i], i)) for i in range(n)])      # per bead: do its two contact patterns agree?
    p_h = compartment_from_hic(hic, smooth=track)
    st = None if state is None else gaussian_filter1d(np.asarray(state, float), STATE_SMOOTH, mode="nearest")      # sliding-window smoothing
    p_m, r_pc = (st, _r(p_h, st)) if st is not None else (pc1(m, smooth=track)[0], abs(_r(p_h, pc1(m, smooth=track)[0])))
    return dict(r_log=_r(np.log10(h[ok]), np.log10(m[ok])), r_oe=_r(np.log2(oe_h[ok] + 1e-3), np.log2(oe_m[ok] + 1e-3)),
                scc=_scc(np.where(seen, h, np.nan), m), r_cm=_r(c_h[iu & seen], c_m[iu & seen]), corr_h=c_h, corr_m=c_m, row_r=np.where(seen.any(0), row_r, np.nan), r_ps=_r(np.log(ps_h), np.log(ps_m)), r_ins=_r(ins_h, ins_m),
                f1=float(f1) if np.isfinite(f1) else NAN, r_pc=r_pc, n_pairs=int(ok.sum()),
                ps_data=ps_h, ps_model=ps_m, ins_data=ins_h, ins_model=ins_m, b_data=b_h + window, b_model=b_m + window,
                pc_data=p_h, pc_model=p_m * (1 if state is not None or _r(p_h, p_m) >= 0 else -1), window=window, h=h, m=m)
