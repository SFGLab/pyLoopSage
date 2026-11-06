"""Chain crossings and the strand-crossing barrier of the excluded volume (EV). Pure numpy, no MD engine needed.

A crossing is two non-adjacent bonds (segments) passing through each other between two snapshots of the chain.
"""
import numpy as np
from scipy.spatial import cKDTree

KB = 0.008314462618                                   # kJ/mol/K
_V = np.linalg.inv(np.vander([0, 1 / 3, 2 / 3, 1], 4))    # samples of a cubic at 4 times -> its coefficients


def _det(a, b, c):
    return np.einsum("...i,...i->...", a, np.cross(b, c))


def _cubic_roots(f):
    """Roots in [0, 1] of the cubic through the 4 samples f (t = 0, 1/3, 2/3, 1)."""
    r = np.roots(_V @ f)
    return [float(x.real) for x in r if abs(x.imag) < 1e-9 and 0 <= x.real <= 1]


def crossings(prev, cur, min_sep: int = 2):
    """Crossings between two snapshots (n, 3): list of (segment i, segment j, sign); segment i joins beads i and i+1.

    Beads move linearly between the snapshots. Two segments cross when the four end points are coplanar at some time
    and the segments intersect there; the sign is the direction in which the signed volume changes.
    """
    prev, cur = np.asarray(prev, float), np.asarray(cur, float)
    mid0, mid1 = (prev[1:] + prev[:-1]) / 2, (cur[1:] + cur[:-1]) / 2
    length = max(np.linalg.norm(np.diff(prev, axis=0), axis=1).max(), np.linalg.norm(np.diff(cur, axis=0), axis=1).max())
    reach = length + 2 * np.linalg.norm(cur - prev, axis=1).max() + 1e-9
    pairs = cKDTree(mid0).query_pairs(reach, output_type="ndarray")
    pairs = np.unique(np.vstack([pairs, cKDTree(mid1).query_pairs(reach, output_type="ndarray")]), axis=0)
    pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) >= min_sep]
    if not len(pairs):
        return []
    i, j = pairs[:, 0], pairs[:, 1]
    pos = lambda t: prev + t * (cur - prev)
    f = np.empty((len(pairs), 4))
    for k, t in enumerate((0, 1 / 3, 2 / 3, 1)):
        p = pos(t)
        a0, a1, b0, b1 = p[i], p[i + 1], p[j], p[j + 1]
        f[:, k] = _det(a1 - a0, b0 - a0, b1 - a0)
    out = []
    for q in np.flatnonzero((f.min(1) < 0) & (f.max(1) > 0)):     # a sign change: the segments become coplanar
        for t in _cubic_roots(f[q]):
            p = pos(t)
            a0, a1, b0, b1 = p[i[q]], p[i[q] + 1], p[j[q]], p[j[q] + 1]
            d1, d2, w = a1 - a0, b1 - b0, b0 - a0
            n = np.cross(d1, d2)
            den = float(n @ n)
            if den < 1e-18:
                continue
            s, u = float(np.cross(w, d2) @ n) / den, float(np.cross(w, d1) @ n) / den
            if 0 <= s <= 1 and 0 <= u <= 1:
                out.append((int(i[q]), int(j[q]), int(np.sign(f[q, -1] - f[q, 0])) or 1))
    return out


def barrier_kT(strength: float, power: float, temperature: float = 310.0, bond: float = 0.1, eps: float = 0.05, r_small: float = 0.1) -> float:
    """Lowest EV energy (in kT) at which two bonds intersect: the barrier that keeps strands from passing through each other.

    Four bead pairs repel with E(r) = (eps^2 strength / (r + r_small))^power; the crossing point and angle are chosen to minimise the
    sum (a lower bound: further neighbours add a little).
    """
    E = lambda r: (eps * eps * strength / (r + r_small)) ** power
    best = np.inf
    for s in np.linspace(0, 1, 21):
        for t in np.linspace(0, 1, 21):
            for th in np.linspace(np.pi / 12, np.pi / 2, 8):
                a = (np.array([-s * bond, 0, 0]), np.array([(1 - s) * bond, 0, 0]))
                d = np.array([np.cos(th), np.sin(th), 0])
                b = (-t * bond * d, (1 - t) * bond * d)
                best = min(best, sum(E(np.linalg.norm(x - y)) for x in a for y in b))
    return float(best / (KB * temperature))


def strength_for_barrier(target_kT: float, strength: float, power: float, temperature: float = 310.0, **kw) -> float:
    """EV strength that gives the target barrier (the barrier grows as strength**power)."""
    return float(strength * (target_kT / barrier_kT(strength, power, temperature, **kw)) ** (1 / power))
