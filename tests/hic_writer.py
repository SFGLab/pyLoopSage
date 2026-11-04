"""Minimal .hic (v8) writer for tests: one chromosome, raw counts, one BP resolution (no normalisation vectors)."""
import struct
import zlib

import numpy as np


def _s(x: str) -> bytes:
    return x.encode() + b"\0"


def write_hic(path, mat: np.ndarray, chrom="chr1", res=10000, genome="test"):
    """Write the upper triangle of an integer count matrix (bins of `res` bp, starting at 0) as a .hic file."""
    n = mat.shape[0]
    size = n * res
    ii, jj = np.nonzero(np.triu(mat) > 0)
    rows = {}
    for i, j in zip(ii, jj):                              # records (binX = i, binY = j) grouped by binY
        rows.setdefault(j, []).append((i, float(mat[i, j])))
    body = struct.pack("<iii", len(ii), 0, 0) + struct.pack("<B", 1) + struct.pack("<B", 1)   # float counts, list-of-rows (v8: 16-bit positions)
    body += struct.pack("<h", len(rows))
    for y in sorted(rows):
        body += struct.pack("<hh", y, len(rows[y]))
        for x, v in rows[y]:
            body += struct.pack("<hf", x, v)
    block = zlib.compress(body)
    chroms = [("All", size // 1000), (chrom, size)]
    head = b"HIC\0" + struct.pack("<i", 8)
    pos_master = len(head) + 8                             # filled below
    attrs = struct.pack("<i", 0)
    chr_bytes = struct.pack("<i", len(chroms)) + b"".join(_s(c) + struct.pack("<i", l) for c, l in chroms)
    res_bytes = struct.pack("<i", 1) + struct.pack("<i", res) + struct.pack("<i", 0)
    header_len = len(head) + 8 + len(_s(genome)) + len(attrs) + len(chr_bytes) + len(res_bytes)
    # matrix record sits right after the header, the block after the matrix metadata
    meta = struct.pack("<iii", 1, 1, 1) + _s("BP") + struct.pack("<i", 0) + struct.pack("<ffff", float(mat.sum()), float(len(ii)), 1.0, 1.0)
    meta += struct.pack("<iiii", res, n + 1, 1, 1)
    meta_len = len(meta) + 4 + 8 + 4
    block_pos = header_len + meta_len
    meta += struct.pack("<iqi", 0, block_pos, len(block))
    matrix_pos, matrix_size = header_len, len(meta)
    master_pos = block_pos + len(block)
    key = _s("1_1")
    entries = struct.pack("<i", 1) + key + struct.pack("<qi", matrix_pos, matrix_size)
    footer = struct.pack("<i", len(entries)) + entries + struct.pack("<iii", 0, 0, 0)   # no expected / normalised-expected / norm vectors
    with open(path, "wb") as f:
        f.write(head + struct.pack("<q", master_pos) + _s(genome) + attrs + chr_bytes + res_bytes)
        f.write(meta)
        f.write(block)
        f.write(footer)
    return path
