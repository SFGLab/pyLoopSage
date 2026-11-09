"""Position weight matrix (PWM) from a PFM and a fast two-strand scan.

Score of a window = sum over positions of log2( P(base | motif) / 0.25 ); higher = closer to the consensus.
A hit has a forward-genome start and a strand: '+' = motif as written, '-' = its reverse complement.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from loopsage.motif_finder.fasta import encode

DEFAULT_PFM = Path(__file__).parent / "data" / "MA0139.1.pfm"     # JASPAR CTCF, 19 bp


def read_counts(path) -> np.ndarray:
    """(4, L) counts, rows A C G T; accepts plain 4-row files and JASPAR 'A [ ... ]' lines."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">") or not line.strip():
            continue
        vals = line.replace("[", " ").replace("]", " ").split()
        if vals and vals[0] in ("A", "C", "G", "T"):
            vals = vals[1:]
        rows.append([float(v) for v in vals])
    if len(rows) != 4 or len({len(r) for r in rows}) != 1:
        raise ValueError(f"{path}: expected 4 rows (A, C, G, T) of equal length")
    return np.array(rows)


@dataclass
class Pssm:
    M: np.ndarray            # (4, L) log2-odds
    name: str = ""

    @property
    def length(self) -> int:
        return self.M.shape[1]

    @property
    def max_score(self) -> float:
        return float(self.M.max(axis=0).sum())

    @property
    def min_score(self) -> float:
        return float(self.M.min(axis=0).sum())

    def cutoff(self, pvalue: float, res: int = 100) -> float:
        """Smallest score whose chance in random sequence (uniform bases) is <= pvalue (exact, scores binned to 1/res)."""
        q = np.round(self.M * res).astype(int)
        dist = np.ones(1)
        for i in range(self.length):                           # distribution of the summed score, one column at a time
            lo = q[:, i].min()
            new = np.zeros(len(dist) + q[:, i].max() - lo)
            for b in range(4):
                new[q[b, i] - lo:q[b, i] - lo + len(dist)] += 0.25 * dist
            dist = new
        tail = np.cumsum(dist[::-1])[::-1]                     # P(score >= x)
        k = int(np.argmax(tail <= pvalue))
        return (q.min(axis=0).sum() + k) / res

    def consensus(self) -> str:
        return "".join("ACGT"[i] for i in self.M.argmax(axis=0))

    def _windows(self, codes, M):
        n = len(codes) - self.length + 1
        ext = np.vstack([M, np.full(self.length, -1e3)])      # N inside a window kills the hit
        s = np.zeros(n)
        for i in range(self.length):
            s += ext[codes[i:i + n], i]
        return s

    def scan(self, seq: str, cutoff: float):
        """All hits >= cutoff on both strands: (start, strand, score) arrays; start is on the forward sequence."""
        if len(seq) < self.length:
            return np.array([], int), np.array([], int), np.array([])
        codes = encode(seq)
        rc = self.M[::-1, ::-1]                                # reverse complement: swap A<->T, C<->G, flip positions
        starts, strands, scores = [], [], []
        for strand, M in ((1, self.M), (-1, rc)):
            s = self._windows(codes, M)
            i = np.flatnonzero(s >= cutoff)
            starts.append(i); strands.append(np.full(len(i), strand)); scores.append(s[i])
        return np.concatenate(starts), np.concatenate(strands), np.concatenate(scores)


def load_pfm(path=None, pseudocount: float = 0.8) -> Pssm:
    """PFM file -> Pssm. Pseudocounts avoid -inf scores for bases never seen at a position."""
    path = Path(path) if path else DEFAULT_PFM
    counts = read_counts(path)
    p = (counts + pseudocount * 0.25) / (counts.sum(axis=0) + pseudocount)
    return Pssm(np.log2(p / 0.25), path.stem)
