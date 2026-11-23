"""Agreement of the maps of independent replicas (loop contact maps, MD contact and distance maps)."""
import numpy as np

from loopsage.analysis.maps import pearson_maps


def map_spread(maps):
    """(mean, std) over replicas of a list of equally sized matrices."""
    a = np.asarray(maps, float)
    return a.mean(axis=0), a.std(axis=0)


def map_correlation(maps) -> np.ndarray:
    """(R, R) Pearson r between the upper triangles of every pair of replica maps."""
    r = np.eye(len(maps))
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            r[i, j] = r[j, i] = pearson_maps(maps[i], maps[j])
    return r


def mean_offdiag(corr) -> float:
    """Mean of the off-diagonal entries of a correlation matrix (nan for one replica)."""
    iu = np.triu_indices(len(corr), 1)
    return float(np.mean(corr[iu])) if len(iu[0]) else float("nan")
