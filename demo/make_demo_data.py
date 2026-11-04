"""Generate a small synthetic BEDPE of CTCF-like loops (for the demos and tests)."""
from pathlib import Path

import numpy as np

REGION = ("chr1", 77_000_000, 95_000_000)


def _orientations(rng):
    """P(reverse) of the two anchors: mostly convergent, some tandem/divergent, some weak or unknown calls."""
    kind = rng.choice(["conv", "tandem", "div", "weak", "unknown"], p=[0.6, 0.12, 0.08, 0.12, 0.08])
    sharp = lambda forward: f"{rng.beta(1, 8) if forward else rng.beta(8, 1):.2f}"
    if kind == "conv":
        return sharp(True), sharp(False)
    if kind == "tandem":
        f = bool(rng.integers(2))
        return sharp(f), sharp(f)
    if kind == "div":
        return sharp(False), sharp(True)
    if kind == "weak":
        return f"{rng.uniform(0.4, 0.6):.2f}", f"{rng.uniform(0.4, 0.6):.2f}"
    return "-1", "-1"


def write_bedpe(path, n_loops: int = 60, seed: int = 0, region=REGION) -> Path:
    """Loops cluster around a few TAD-like domains; columns follow the BEDPE + score/orientation layout."""
    chrom, start, end = region
    rng = np.random.default_rng(seed)
    centers = rng.uniform(start + 1e6, end - 1e6, 8)            # domain centres
    rows = []
    for _ in range(n_loops):
        c = rng.choice(centers)
        a = int(c + rng.normal(0, 0.6e6)) // 5000 * 5000
        b = a + int(rng.uniform(2e5, 1.2e6)) // 5000 * 5000
        if not (start < a and b < end - 5000):
            continue
        p1, p2 = _orientations(rng)
        rows.append(f"{chrom}\t{a}\t{a + 5000}\t{chrom}\t{b}\t{b + 5000}\t{int(rng.integers(5, 40))}\t{p1}\t{p2}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")
    return path


def write_compartments(path, seed: int = 2, region=REGION) -> Path:
    """Alternating A/B compartment BED (score > 0.5 means A)."""
    chrom, start, end = region
    rng, rows, pos, k = np.random.default_rng(seed), [], start, 0
    while pos < end:
        stop = min(pos + int(rng.uniform(0.5e6, 2.2e6)) // 50_000 * 50_000, end)
        score = rng.uniform(0.6, 0.95) if k % 2 == 0 else rng.uniform(0.05, 0.4)
        rows.append(f"{chrom}\t{pos}\t{stop}\t{'A' if k % 2 == 0 else 'B'}\t{score:.2f}\t.")
        pos, k = stop, k + 1
    Path(path).write_text("\n".join(rows) + "\n")
    return Path(path)


def write_bigwigs(bedpe, ctcf_path, lef_path, seed: int = 3, region=REGION):
    """CTCF-like ChIP track (peaks at anchors) and a smoother LEF-loading track; needs pyBigWig."""
    import pyBigWig
    chrom, start, end = region
    rng, step = np.random.default_rng(seed), 5000
    pos = np.arange(start, end, step)
    anchors = np.array([float(l.split("\t")[i]) for l in Path(bedpe).read_text().splitlines() for i in (1, 4)])
    ctcf = np.zeros(len(pos))
    for a in anchors:
        ctcf += np.exp(-0.5 * ((pos - a) / 15_000) ** 2) * rng.uniform(2, 6)
    lef = np.convolve(ctcf, np.ones(25) / 25, "same") + rng.uniform(0, 0.3, len(pos))
    for path, vals in ((ctcf_path, ctcf + rng.uniform(0, 0.2, len(pos))), (lef_path, lef)):
        bw = pyBigWig.open(str(path), "w")
        bw.addHeader([(chrom, 248_956_422)])
        bw.addEntries(chrom, [int(p) for p in pos], values=[float(v) for v in vals], span=step, step=step)
        bw.close()


if __name__ == "__main__":
    d = Path(__file__).parent / "data"
    out = write_bedpe(d / "demo_loops.bedpe")
    print(f"wrote {out} ({len(out.read_text().splitlines())} loops)")
    print(f"wrote {write_compartments(d / 'demo_compartments.bed')}")
    try:
        write_bigwigs(out, d / "demo_ctcf.bw", d / "demo_lef.bw")
        print("wrote demo_ctcf.bw and demo_lef.bw")
    except ImportError:
        print("pyBigWig not installed: skipped the BigWig tracks")
