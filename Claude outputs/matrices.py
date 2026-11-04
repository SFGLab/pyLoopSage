"""Matrix helpers for contact maps."""
import numpy as np


def minmax(x) -> np.ndarray:
    """Scale to [0, 1]."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def ps_curve(mat) -> np.ndarray:
    """Mean contact vs. bead separation s (diagonal decay, P(s))."""
    return np.array([np.diagonal(mat, k).mean() for k in range(mat.shape[0])])


def insulation_score(mat, window: int = 5) -> np.ndarray:
    """Diamond insulation score: mean contact across each bead."""
    n = mat.shape[0]
    ins = np.zeros(n)
    for i in range(window, n - window):
        ins[i] = mat[i - window:i, i + 1:i + window + 1].mean()
    return ins


def pearson_maps(a, b) -> float:
    """Pearson r between the upper triangles (k>=1) of two matrices."""
    iu = np.triu_indices(a.shape[0], k=1)
    return float(np.corrcoef(a[iu], b[iu])[0, 1])
