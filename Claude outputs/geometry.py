"""Mapping between genomic coordinates (bp) and simulation beads."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    chrom: str
    start: int      # bp
    end: int        # bp
    n_beads: int    # simulation beads

    @property
    def length_bp(self) -> int:
        return self.end - self.start

    @property
    def bp_per_bead(self) -> int:
        return max(self.length_bp // self.n_beads, 1)

    @property
    def label(self) -> str:
        return f"{self.chrom}:{self.start:,}-{self.end:,}"

    def bead_of(self, bp):
        """Bead index of genomic position(s) in bp (clipped to the region)."""
        return np.clip((np.asarray(bp) - self.start) // self.bp_per_bead, 0, self.n_beads - 1)

    def axis_mb(self) -> np.ndarray:
        """Bead-centre genomic positions in Mb (for plot axes)."""
        return (self.start + (np.arange(self.n_beads) + 0.5) * self.bp_per_bead) / 1e6

    def extent_mb(self):
        """(left, right, bottom, top) in Mb for imshow with origin='upper'."""
        s, e = self.start / 1e6, (self.start + self.n_beads * self.bp_per_bead) / 1e6
        return [s, e, e, s]

    def beads_to_bp(self, n):
        return np.asarray(n) * self.bp_per_bead
