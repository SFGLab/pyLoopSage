"""Indexed FASTA reader: random access to a region without loading the genome."""
from pathlib import Path

import numpy as np


def norm_chrom(name) -> str:
    """'1' -> 'chr1', 'MT' -> 'chrM'."""
    s = str(name).strip()
    s = s if s.lower().startswith("chr") else "chr" + s
    return "chrM" if s.lower() in ("chrmt", "chrm") else s


def _build_index(path) -> dict:
    """Scan the file once: {name: (length, offset, bases per line, bytes per line)}."""
    idx, name, length, off, lb, lw = {}, None, 0, 0, 0, 0
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            if line.startswith(b">"):
                if name is not None:
                    idx[name] = (length, off, lb, lw)
                name, length, off, lb, lw = line[1:].split()[0].decode(), 0, pos + len(line), 0, 0
            else:
                n = len(line.rstrip(b"\r\n"))
                if lb == 0 and n:
                    lb, lw = n, len(line)
                length += n
            pos += len(line)
    if name is not None:
        idx[name] = (length, off, lb, lw)
    return idx


def _read_fai(path) -> dict:
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        out[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    return out


class Fasta:
    """fetch(chrom, start, end) -> upper-case string (0-based, half-open); builds / reuses a .fai next to the file."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"reference genome not found: {path}")
        if self.path.suffix.lower() == ".gz":
            raise ValueError("compressed FASTA is not supported: decompress it first (gunzip)")
        fai = Path(str(self.path) + ".fai")
        if fai.is_file():
            self.index = _read_fai(fai)
        else:
            self.index = _build_index(self.path)
            try:                                              # cache the index when the folder is writable
                fai.write_text("".join(f"{k}\t{v[0]}\t{v[1]}\t{v[2]}\t{v[3]}\n" for k, v in self.index.items()))
            except OSError:
                pass
        self.alias = {}
        for k in self.index:                                  # '1' and 'chr1' both resolve
            self.alias.setdefault(norm_chrom(k), k)
        self._f = open(self.path, "rb")

    def has(self, chrom) -> bool:
        return norm_chrom(chrom) in self.alias

    def length(self, chrom) -> int:
        return self.index[self.alias[norm_chrom(chrom)]][0]

    def fetch(self, chrom, start, end) -> str:
        name = self.alias.get(norm_chrom(chrom))
        if name is None:
            raise KeyError(chrom)
        length, off, lb, lw = self.index[name]
        start, end = max(int(start), 0), min(int(end), length)
        if end <= start or lb == 0:
            return ""
        a = off + start // lb * lw + start % lb               # byte of the first and last base
        b = off + (end - 1) // lb * lw + (end - 1) % lb + 1
        self._f.seek(a)
        return self._f.read(b - a).replace(b"\n", b"").replace(b"\r", b"").decode().upper()

    def close(self):
        self._f.close()


def encode(seq: str) -> np.ndarray:
    """A,C,G,T -> 0..3, anything else (N, ...) -> 4."""
    lut = np.full(256, 4, dtype=np.int8)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = lut[ord(c.lower())] = i
    return lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
