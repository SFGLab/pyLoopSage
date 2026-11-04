"""Genome constants and default paths.

- Human-readable formatting with explicit units.
- Mapping between genomic coordinates (bp) and simulation beads.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---- Genome constants and default paths ------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FORCEFIELD = str(SRC_DIR / "molecular_dynamics" / "forcefields" / "classic_sm_ff.xml")
DEFAULT_RESOLUTION_BP = 2000  # bp per bead when n_beads is not given

_NAMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
_HG38 = [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973,
         145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718,
         101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983,
         50818468, 156040895, 57227415, 16569]
_HG19 = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663,
         146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540,
         102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895,
         51304566, 155270560, 59373566, 16571]

# chromosome sizes in bp, per genome assembly
CHROM_LENGTHS = {"hg38": dict(zip(_NAMES, _HG38)), "hg19": dict(zip(_NAMES, _HG19))}


# ---- Human-readable formatting with explicit units -------------------------

def fmt_bp(bp: float) -> str:
    bp = float(bp)
    if abs(bp) >= 1e6:
        return f"{bp / 1e6:.2f} Mb"
    if abs(bp) >= 1e3:
        return f"{bp / 1e3:.1f} kb"
    return f"{bp:.0f} bp"


def fmt_num(x, digits: int = 4) -> str:
    """Compact number: integers stay integers, floats get significant digits."""
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e9):
        return f"{int(x):,}"
    return f"{x:.{digits}g}"


def fmt_duration(seconds: float) -> str:
    s = float(seconds)
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{int(s // 60)} min {int(s % 60)} s"
    return f"{int(s // 3600)} h {int(s % 3600 // 60)} min"


# ---- Mapping between genomic coordinates (bp) and simulation beads ---------

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
