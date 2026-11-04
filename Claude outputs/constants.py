"""Genome constants and default paths."""
from pathlib import Path

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
