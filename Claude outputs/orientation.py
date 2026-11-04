"""Diagnostics using CTCF motif orientation stored in the input file."""
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr

from preproc.readers import numeric_col, valid_prob
from validation.signals import _beads, bedpe_pairs


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
