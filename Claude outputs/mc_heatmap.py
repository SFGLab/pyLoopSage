"""Hi-C-like contact map averaged over the loop ensemble (no 3D needed)."""
import numpy as np

from common.progress import stage_bar


def stochastic_map(res, n_beads: int, chunk: int = 2048) -> np.ndarray:
    """Each loop contributes a smooth 'tanh' block over its interior; averaged over snapshots.

    A bead pair (i, j) gets weight w_i * w_j, with w_b = tanh((b - left)/s) * tanh((right - b)/s)
    clipped at 0 and s = 5 % of the loop length (at least 1 bead).
    """
    ms, ns = res.production(res.Ms), res.production(res.Ns)
    left, right = np.minimum(ms, ns).ravel(), np.maximum(ms, ns).ravel()
    b, mat = np.arange(n_beads, dtype=np.float32), np.zeros((n_beads, n_beads))
    with stage_bar("Contact map from loops", len(left)) as advance:
        for s in range(0, len(left), chunk):
            lc, rc = left[s:s + chunk, None].astype(np.float32), right[s:s + chunk, None].astype(np.float32)
            sharp = np.maximum((rc - lc) * 0.05, 1.0)
            w = np.clip(np.tanh((b - lc + 0.5) / sharp) * np.tanh((rc - b + 0.5) / sharp), 0, None)
            mat += w.T @ w
            advance(len(lc))
    return mat / ms.shape[1]
