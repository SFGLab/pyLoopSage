"""Read tab-separated BEDPE / BED / narrowPeak tables restricted to the region.

- Signal scaling shared by all loop/peak readers.
- BigWig signal tracks -> one value per bead.
- BEDPE loops -> anchor profiles (L, R), adjacency J and weighted map J_loss.
- BED / narrowPeak CTCF sites -> anchor profiles (L, R).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from common.utils import describe
from logger import get_logger


# ---- Read tab-separated BEDPE / BED / narrowPeak tables restricted to the region ----


@dataclass
class LoopData:
    """Anchor profiles and loop maps at bead resolution."""
    L: np.ndarray             # left-anchor (forward-CTCF-like) binding profile, per bead
    R: np.ndarray             # right-anchor binding profile, per bead
    J: np.ndarray             # N x N binary adjacency (loops + backbone)
    J_loss: np.ndarray        # N x N weighted loop map used by the 2D data energy
    n_sites: int              # anchors (bedpe: 2 x loops) or peaks
    loop_lengths: np.ndarray  # unique loop lengths, beads (empty for peak files)
    stats: dict = field(default_factory=dict)
    fmt: str = "bedpe"


def read_region_table(path, geom, paired: bool) -> pd.DataFrame:
    """Rows fully inside the region on the modelled chromosome (paired: both anchors)."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    keep = (df[0] == geom.chrom)
    for c in (1, 2) + ((4, 5) if paired else ()):
        keep &= (df[c] >= geom.start) & (df[c] < geom.end)
    return df[keep].reset_index(drop=True)


def numeric_col(df, col: int, default: float) -> np.ndarray:
    """Column as floats; missing column or bad cells become `default`."""
    if col >= df.shape[1]:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)


def valid_prob(p) -> bool:
    """True if every value is -1 (unknown) or a probability in [0, 1]."""
    p = np.asarray(p, dtype=float)
    return bool(np.all(np.isfinite(p) & ((p == -1) | ((p >= 0) & (p <= 1)))))


# ---- Signal scaling shared by all loop/peak readers ------------------------

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


# ---- BigWig signal tracks -> one value per bead ----------------------------

def bigwig_to_beads(path, geom, roll: bool = False, scale_pm1: bool = False) -> np.ndarray:
    """Mean signal per bead; roll = blur +/-3 beads, scale_pm1 = rescale to [-1, 1]."""
    import pyBigWig                                   # optional dependency, imported lazily
    bw = pyBigWig.open(str(path))
    try:
        if geom.chrom not in bw.chroms():
            raise ValueError(f"{geom.chrom} not found in {path}")
        raw = np.nan_to_num(np.array(bw.values(geom.chrom, geom.start, geom.end), dtype=float))
    finally:
        bw.close()
    step, n = geom.bp_per_bead, geom.n_beads
    usable = min(len(raw) // step, n)
    w = np.zeros(n)
    w[:usable] = raw[:usable * step].reshape(usable, step).mean(axis=1)
    if roll:
        w = (np.roll(w, 3) + np.roll(w, -3)) / 2
    if scale_pm1:
        lo, hi = w.min(), w.max()
        w = 2 * (w - lo) / (hi - lo) - 1 if hi > lo else np.zeros(n)
    return w


# ---- BEDPE loops -> anchor profiles (L, R), adjacency J and weighted map J_loss ----

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


def vectors_from_bedpe(path, geom, smooth=False, contrastive=True, sigma=1.0, loss_sigma=0.0) -> LoopData:
    """Build model inputs from a BEDPE file for the region in `geom` (loss_sigma > 0: J_loss becomes a smooth field)."""
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
    if loss_sigma > 0:                               # smooth 2D landscape for the loop-map data energy
        J_loss = gaussian_filter(J_loss, loss_sigma)
    L, R = mean_scale(L), mean_scale(R)
    if contrastive:
        J_loss, L, R = contrast_map(J_loss), contrast(L), contrast(R)

    lengths, stats = _stats(J, (x + y) // 2)
    log.step("Input: %d unique loops (%d anchors) in %s", stats["n_loops"], stats["n_anchors"], geom.label)
    return LoopData(L, R, J, np.sqrt(stats["n_anchors"]) * J_loss, stats["n_anchors"], lengths, stats, "bedpe")


# ---- BED / narrowPeak CTCF sites -> anchor profiles (L, R) -----------------


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
