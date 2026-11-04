"""BED / narrowPeak CTCF sites -> anchor profiles (L, R).

Single sites carry no loop pairing, so J is backbone-only and J_loss is zero.
Optional orientation probabilities (from the motif finders) split each site's
weight between L (reverse-biased) and R (forward-biased).
"""
import numpy as np
import pandas as pd

from helpers.stats import describe
from logger import get_logger
from preproc.normalize import contrast, mean_scale, smooth_sparse
from preproc.readers import numeric_col, read_region_table, valid_prob
from preproc.types import LoopData

log = get_logger(__name__)


def _profiles(beads, w, prob, spans, geom, smooth, sigma, contrastive):
    """Accumulate site weights into L/R (optionally spread over each peak's width)."""
    n = geom.n_beads
    known = prob >= 0
    wl, wr = np.where(known, w * (1 - prob), w * 0.5), np.where(known, w * prob, w * 0.5)
    L, R = np.zeros(n), np.zeros(n)
    if spans is None:
        np.add.at(L, beads, wl); np.add.at(R, beads, wr)
    else:
        for (b0, b1), a, b in zip(spans, wl, wr):
            width = max(b1 - b0, 1)
            L[b0:b1] += a / width; R[b0:b1] += b / width
    if smooth:
        L, R = smooth_sparse(L, sigma), smooth_sparse(R, sigma)
    L, R = mean_scale(L), mean_scale(R)
    if contrastive:
        L, R = contrast(L), contrast(R)
    return 2 * L, 2 * R


def _finish(L, R, w, prob, geom, label):
    n = geom.n_beads
    J = np.zeros((n, n))
    idx = np.arange(n - 1)
    J[idx, idx + 1] = J[idx + 1, idx] = 1.0
    stats = dict(n_peaks=int(len(w)), n_with_orientation=int((prob >= 0).sum()), weight=describe(w))
    log.info("%s: %d sites, %d with known orientation", label, len(w), stats["n_with_orientation"])
    return LoopData(L, R, J, np.zeros((n, n)), len(w), np.array([]), stats, label.lower())


def vectors_from_bed(path, geom, smooth=False, contrastive=True, sigma=2.0) -> LoopData:
    """BED: col 5 = score (weight), col 7 = optional reverse-orientation probability."""
    df = read_region_table(path, geom, paired=False)
    beads = np.asarray(geom.bead_of((df[1] + df[2]) // 2))
    w = numeric_col(df, 4, 1.0)
    prob = numeric_col(df, 6, -1.0)
    if not valid_prob(prob):
        log.warning("BED column 7 is not a valid probability: orientation ignored.")
        prob = np.full(len(df), -1.0)
    L, R = _profiles(beads, w, prob, None, geom, smooth, sigma, contrastive)
    return _finish(L, R, w, prob, geom, "BED")


def vectors_from_narrowpeak(path, geom, smooth=False, contrastive=True, sigma=2.0) -> LoopData:
    """narrowPeak: weight = signalValue (fallback score); summit gives the bead."""
    df = read_region_table(path, geom, paired=False)
    if len(df) == 0:
        raise ValueError(f"no narrowPeak entries found in {geom.label}")
    w = numeric_col(df, 6, 0.0)
    if not np.any(w > 0):
        w = numeric_col(df, 4, 1.0)
    offset = numeric_col(df, 9, -1)                  # summit offset from start (-1 = missing)
    mid = ((df[1] + df[2]) // 2).to_numpy()
    beads = np.asarray(geom.bead_of(np.where(offset >= 0, df[1].to_numpy() + offset, mid)))
    b0 = np.clip((df[1].to_numpy() - geom.start) // geom.bp_per_bead, 0, geom.n_beads)
    b1 = np.clip((df[2].to_numpy() - geom.start) // geom.bp_per_bead + 1, 0, geom.n_beads)

    prob = np.full(len(df), -1.0)                    # orientation: appended forward-probability column
    for col in (10, 5):
        cand = numeric_col(df, col, -1.0) if df.shape[1] > col else None
        if cand is not None and valid_prob(cand) and np.any(cand >= 0):
            prob = np.where(cand >= 0, 1 - cand, -1.0)
            break
    L, R = _profiles(beads, w, prob, list(zip(b0, b1)), geom, smooth, sigma, contrastive)
    return _finish(L, R, w, prob, geom, "narrowPeak")
