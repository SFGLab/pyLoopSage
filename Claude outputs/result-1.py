"""Molecular dynamics output container (with save / load)."""
import json
from dataclasses import dataclass, field

import numpy as np

_ARRAYS = ("coords", "avg_map", "std_map", "rg", "epot", "frame_idx")


@dataclass
class MDResult:
    coords: np.ndarray     # (frames, beads, 3) positions in nm (float32)
    avg_map: np.ndarray    # mean contact map over frames
    std_map: np.ndarray    # std of the contact map over frames
    rg: np.ndarray         # radius of gyration per frame, nm
    epot: np.ndarray       # potential energy per frame, kJ/mol
    frame_idx: np.ndarray  # MC snapshot index each frame was built from
    meta: dict = field(default_factory=dict)   # platform, runtime_s, dt_fs, steps per frame...

    def save(self, path) -> None:
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **{k: getattr(self, k) for k in _ARRAYS})

    @classmethod
    def load(cls, path) -> "MDResult":
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in _ARRAYS}, meta=json.loads(str(z["meta"])))
