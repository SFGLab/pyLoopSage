"""Containers for preprocessed model inputs (with disk cache)."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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


@dataclass
class InputData:
    loops: LoopData
    BWs: np.ndarray                  # (n_tracks, N) extra binding tracks
    bw_names: list = field(default_factory=list)
    lef_track: np.ndarray = None     # LEF loading preference per bead
    comp: np.ndarray = None          # compartment signal in [-1, 1] per bead

    def save(self, path) -> None:
        lo = self.loops
        arrays = dict(L=lo.L, R=lo.R, J=lo.J, J_loss=lo.J_loss, lengths=lo.loop_lengths, BWs=self.BWs)
        for name in ("lef_track", "comp"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        meta = json.dumps(dict(n_sites=int(lo.n_sites), stats=lo.stats, fmt=lo.fmt, bw_names=self.bw_names))
        np.savez_compressed(path, meta=np.array(meta), **arrays)

    @classmethod
    def load(cls, path) -> "InputData":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        loops = LoopData(z["L"], z["R"], z["J"], z["J_loss"], meta["n_sites"], z["lengths"], meta["stats"], meta["fmt"])
        return cls(loops, z["BWs"], meta["bw_names"],
                   z["lef_track"] if "lef_track" in z else None, z["comp"] if "comp" in z else None)
