"""Monte Carlo output container (with save / load)."""
import json
from dataclasses import dataclass, field

import numpy as np

_ARRAYS = ("Ms", "Ns", "Es", "Ks", "Fs", "Bs", "ufs", "Ts", "epi", "acc")


@dataclass
class MCResult:
    Ms: np.ndarray    # (n_lef_total, n_samples) left-leg bead of each LEF over time
    Ns: np.ndarray    # right legs
    Es: np.ndarray    # total energy per snapshot (a.u.)
    Ks: np.ndarray    # crossing energy
    Fs: np.ndarray    # folding energy
    Bs: np.ndarray    # binding (data) energy
    ufs: np.ndarray   # unfolded fraction of the chain
    Ts: np.ndarray    # temperature at each snapshot
    epi: np.ndarray   # (n_beads, n_samples) epigenetic states
    acc: np.ndarray   # (3, 2) proposed/accepted for rebind, slide, state flip
    meta: dict = field(default_factory=dict)   # n_beads, n_lef, mc_step, burnin, runtime_s, ...

    @property
    def n_samples(self) -> int:
        return self.Es.shape[0]

    @property
    def burnin_idx(self) -> int:
        """First post-burn-in snapshot."""
        return min(self.meta["burnin"] // self.meta["mc_step"], self.n_samples - 1)

    def production(self, arr):
        """Slice of the last axis after burn-in."""
        return arr[..., self.burnin_idx:]

    def save(self, path) -> None:
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **{k: getattr(self, k) for k in _ARRAYS})

    @classmethod
    def load(cls, path) -> "MCResult":
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in _ARRAYS}, meta=json.loads(str(z["meta"])))
