"""BigWig signal tracks -> one value per bead."""
import numpy as np


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
