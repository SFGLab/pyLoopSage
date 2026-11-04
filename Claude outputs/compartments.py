"""Compartment BED (A/B sub-compartments) -> signal in [-1, 1] per bead."""
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def _label_value(label) -> float:
    """'A.1.2' -> +1.2, 'B.1' -> -1.0: sign from A/B, magnitude grows with depth."""
    if not isinstance(label, str) or label[:1] not in "AB":
        return 0.0
    return (1.0 if label[0] == "A" else -1.0) * (1.0 + 0.2 * (len(label.split(".")) - 1))


def _values(df, use_score: bool) -> np.ndarray:
    """Per-row value: 2*score-1 if the score column is numeric, else from the label."""
    by_label = df[3].map(_label_value).to_numpy(dtype=float)
    if not use_score or df.shape[1] < 5:
        return by_label
    score = pd.to_numeric(df[4], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isnan(score), by_label, 2.0 * score - 1.0)


def load_compartments_bed(path, geom, use_score=True, spline_s=None) -> np.ndarray:
    """Average row values per bead, optional spline smoothing, genome-wide [-1, 1] scaling."""
    full = pd.read_csv(path, sep="\t", header=None)
    vals = _values(full, use_score)
    gmin, gmax = vals.min(), vals.max()              # global reference keeps scale comparable
    sel = (full[0] == geom.chrom) & (full[1] < geom.end) & (full[2] > geom.start)
    if not sel.any():
        raise ValueError(f"no compartments found in {geom.label}")
    df, v = full[sel].reset_index(drop=True), vals[sel.to_numpy()]

    n = geom.n_beads
    signal, count = np.zeros(n), np.zeros(n)
    b0 = geom.bead_of(np.maximum(df[1].to_numpy(), geom.start))
    b1 = geom.bead_of(np.minimum(df[2].to_numpy(), geom.end))
    for lo, hi, val in zip(b0, b1, v):
        signal[lo:hi + 1] += val
        count[lo:hi + 1] += 1
    covered = count > 0
    signal[covered] /= count[covered]

    if spline_s is not None and covered.sum() > 3:   # fill gaps, then smooth
        x = np.arange(n)
        signal[~covered] = np.interp(x[~covered], x[covered], signal[covered])
        signal = UnivariateSpline(x, signal, s=spline_s)(x)
    return 2.0 * (signal - gmin) / (gmax - gmin) - 1.0 if gmax > gmin else np.zeros(n)
