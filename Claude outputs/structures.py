"""Initial polymer shapes. One coordinate unit = 0.1 nm (the bond length)."""
import numpy as np

from logger import get_logger

log = get_logger(__name__)


def _unit_vectors(n):
    v = np.random.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def random_walk(n, step=1.0):
    return np.vstack([np.zeros(3), np.cumsum(step * _unit_vectors(n - 1), axis=0)])


def confined_random_walk(n, box=5):
    """Lattice random walk clipped to a cube (folds the chain in)."""
    V = np.zeros((n, 3))
    for i in range(1, n):
        V[i] = np.clip(V[i - 1] + np.random.choice([-1, 1], size=3), -box, box)
    return V


def self_avoiding_random_walk(n, step=1.0, radius=0.5, eps=1e-3, tries=1000):
    """Random walk whose beads avoid overlap (best effort after `tries` attempts)."""
    pts = np.zeros((n, 3))
    for k in range(1, n):
        for _ in range(tries):
            cand = pts[k - 1] + step * _unit_vectors(1)[0]
            if np.linalg.norm(pts[:k] - cand, axis=1).min() >= 2 * radius - eps:
                break
        pts[k] = cand
    return pts


def circle(n, z_stretch=1.0):
    r = 1 / (2 * np.sin(np.pi / n))                  # unit spacing between neighbours
    a = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.cos(a), r * np.sin(a), z_stretch * np.arange(1, n + 1) / n])


def helix(n, radius=1, pitch=2):
    t = np.linspace(0, 4 * np.pi, n)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), np.linspace(0, pitch * n, n)])


def spiral(n, radius0=1, pitch=1, growth=0.05):
    t, r = np.linspace(0, 4 * np.pi, n), radius0 + growth * np.arange(n)
    return np.column_stack([r * np.cos(t), r * np.sin(t), np.linspace(0, pitch * n, n)])


def sphere(n, radius=1):
    phi, cos_t = np.random.uniform(0, 2 * np.pi, n), np.random.uniform(-1, 1, n)
    r, sin_t = radius * np.random.uniform(0, 1, n) ** (1 / 3), np.sqrt(1 - cos_t ** 2)
    return np.column_stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t])


SHAPES = {"rw": random_walk, "confined_rw": confined_random_walk, "self_avoiding_rw": self_avoiding_random_walk,
          "circle": circle, "helix": helix, "spiral": spiral, "sphere": sphere}


def make_structure(n: int, mode: str = "rw") -> np.ndarray:
    """(n, 3) starting coordinates for the chosen shape."""
    if mode not in SHAPES:
        raise ValueError(f"unknown initial structure '{mode}'; choose from {sorted(SHAPES)}")
    log.info("Initial structure: %s, %d beads", mode, n)
    return SHAPES[mode](n)
