"""BEDPE loops -> anchor profiles (L, R), adjacency J and weighted map J_loss.

Columns: chrom1 s1 e1 chrom2 s2 e2 [weight] [p_rev_anchor1] [p_rev_anchor2],
where p is the probability that the anchor's CTCF motif is reverse-oriented
(-1 = unknown). Orientation splits each anchor's weight between L and R.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from helpers.stats import describe
from logger import get_logger
from preproc.normalize import contrast, contrast_map, mean_scale
from preproc.readers import numeric_col, read_region_table, valid_prob
from preproc.types import LoopData

log = get_logger(__name__)


def _split(w, p):
    """Weights (to L, to R) of one anchor given its reverse-probability p."""
    known = p >= 0
    return np.where(known, w * (1 - p), w * 0.5), np.where(known, w * p, w * 0.5)


def _accumulate(x, y, w, p1, p2, n):
    """Add every loop's weight to L, R and J_loss."""
    L, R, Jl = np.zeros(n), np.zeros(n), np.zeros((n, n))
    if p1 is None:                                   # no orientation: split evenly
        np.add.at(Jl, (x, y), w); np.add.at(Jl, (y, x), w)
        for v in (x, y):
            np.add.at(L, v, w); np.add.at(R, v, w)
    else:
        wx, wy = _split(w, p1)
        wx2, wy2 = _split(w, p2)
        np.add.at(Jl, (x, y), 0.5 * (wx + wy2)); np.add.at(Jl, (y, x), 0.5 * (wy + wx2))
        np.add.at(L, x, wx); np.add.at(R, x, wy); np.add.at(L, y, wx2); np.add.at(R, y, wy2)
    return L, R, (Jl + Jl.T) / 2


def _stats(J, centers):
    """Unique-loop statistics in beads."""
    pairs = np.argwhere(np.triu(J, k=2) > 0)
    lengths = pairs[:, 1] - pairs[:, 0]
    gaps = np.diff(np.sort(centers)) if len(centers) > 1 else np.array([])
    stats = dict(n_loops=int(len(lengths)), n_anchors=int(2 * len(lengths)),
                 loop_length=describe(lengths), distance_between_loops=describe(gaps))
    return lengths, stats


def vectors_from_bedpe(path, geom, smooth=False, contrastive=True, sigma=1.0) -> LoopData:
    """Build model inputs from a BEDPE file for the region in `geom`."""
    n = geom.n_beads
    df = read_region_table(path, geom, paired=True)
    x = np.asarray(geom.bead_of(((df[1] + df[2]) // 2).to_numpy()))
    y = np.asarray(geom.bead_of(((df[4] + df[5]) // 2).to_numpy()))
    keep = x != y                                    # drop loops inside a single bead
    x, y, df = x[keep], y[keep], df[keep].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"no loops with distinct anchors found in {geom.label}")

    p1 = p2 = None
    if df.shape[1] > 8:
        a, b = numeric_col(df, 7, -1), numeric_col(df, 8, -1)
        if valid_prob(a) and valid_prob(b):
            p1, p2 = a, b
        else:
            log.warning("Columns 8/9 are not valid orientation probabilities: ignored.")
    L, R, J_loss = _accumulate(x, y, numeric_col(df, 6, 1.0), p1, p2, n)

    J = np.zeros((n, n))
    J[x, y] = J[y, x] = 1.0
    idx = np.arange(n - 1)
    J[idx, idx + 1] = J[idx + 1, idx] = 1.0          # chain backbone

    if smooth:
        L, R = gaussian_filter1d(L, sigma, mode="nearest"), gaussian_filter1d(R, sigma, mode="nearest")
        J_loss = gaussian_filter(J_loss, sigma)
    L, R = mean_scale(L), mean_scale(R)
    if contrastive:
        J_loss, L, R = contrast_map(J_loss), contrast(L), contrast(R)

    lengths, stats = _stats(J, (x + y) // 2)
    log.info("BEDPE: %d loops (%d anchors) in %s", stats["n_loops"], stats["n_anchors"], geom.label)
    return LoopData(L, R, J, np.sqrt(stats["n_anchors"]) * J_loss, stats["n_anchors"], lengths, stats, "bedpe")
