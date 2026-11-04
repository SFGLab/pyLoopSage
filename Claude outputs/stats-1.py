"""Basic statistics used across stages."""
import numpy as np
from scipy.stats import norm


def describe(x) -> dict:
    """mean / median / min / max / std of an array (zeros if empty)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return dict(mean=0.0, median=0.0, min=0.0, max=0.0, std=0.0)
    return dict(mean=x.mean(), median=np.median(x), min=x.min(), max=x.max(), std=x.std())


def pearson_ci(r: float, n: int, alpha: float = 0.05):
    """95% CI of a Pearson r (Fisher z-transform)."""
    if n < 4 or abs(r) >= 1:
        return np.nan, np.nan
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(n - 3)
    crit = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def strength_label(r: float) -> str:
    """Plain-language label for |r| (Cohen-style thresholds)."""
    a = abs(r)
    for cut, name in ((0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"), (0.7, "strong")):
        if a < cut:
            return name
    return "very strong"


def normalize_signal(vec, log_transform: bool = True) -> np.ndarray:
    """Optional log1p (tames heavy tails), then z-score."""
    v = np.asarray(vec, dtype=float)
    if log_transform:
        v = np.log1p(np.clip(v, 0, None))
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()
