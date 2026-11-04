"""Distances -> contact probabilities (Hi-C-like maps) from 3D coordinates."""
import numpy as np
from scipy.spatial.distance import pdist, squareform

# contact kernels of distance d (nm) and contact radius r (nm)
KERNELS = {
    "gaussian": lambda d, r: np.exp(-d ** 2 / (2 * r ** 2)),
    "soft": lambda d, r: 1.0 / (1.0 + (d / r) ** 3),
    "binary": lambda d, r: (d <= r).astype(float),
    "exponential": lambda d, r: np.exp(-d / r),
    "power_law": lambda d, r: (d / r + 1e-3) ** -3.0,
}


def contact_map(coords, radius: float = 1.0, mode: str = "gaussian") -> np.ndarray:
    """N x N contact map of one frame (coords in nm)."""
    if not np.all(np.isfinite(coords)):
        raise ValueError("non-finite coordinates: the MD became unstable")
    return KERNELS[mode](squareform(pdist(coords)), radius)


class RunningMean:
    """Streaming mean and std of matrices (no need to keep every frame)."""

    def __init__(self):
        self.n, self.s, self.s2 = 0, None, None

    def add(self, x):
        if self.s is None:
            self.s, self.s2 = np.zeros_like(x, float), np.zeros_like(x, float)
        self.n += 1
        self.s += x
        self.s2 += x * x

    @property
    def mean(self):
        return self.s / self.n

    @property
    def std(self):
        return np.sqrt(np.maximum(self.s2 / self.n - self.mean ** 2, 0.0))
