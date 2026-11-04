"""Signal scaling shared by all loop/peak readers."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

_EPS = 1e-6


def mean_scale(v) -> np.ndarray:
    """Divide by the mean (so average bead signal = 1)."""
    m = np.mean(v)
    return v / m if m != 0 else v


def contrast(v) -> np.ndarray:
    """z-score then tanh: bounded signal in (-1, 1) with loop-free regions < 0."""
    return np.tanh((v - np.mean(v)) / (np.std(v) + _EPS))


def contrast_map(m) -> np.ndarray:
    """Same contrast for a 2D map, after dividing by its maximum."""
    m = m / (np.max(m) + _EPS)
    return np.tanh((m - np.mean(m)) / (np.std(m) + _EPS))


def smooth_sparse(v, sigma: float) -> np.ndarray:
    """Gaussian smoothing, with sigma capped so sparse peaks are not erased."""
    nz = np.flatnonzero(v > 0)
    cap = max(3.0, 3.0 * np.median(np.diff(nz))) if len(nz) > 2 else 5.0
    return gaussian_filter1d(v, sigma=min(sigma, cap), mode="nearest")
