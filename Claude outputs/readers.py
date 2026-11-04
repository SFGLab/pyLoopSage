"""Read tab-separated BEDPE / BED / narrowPeak tables restricted to the region."""
import numpy as np
import pandas as pd


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
