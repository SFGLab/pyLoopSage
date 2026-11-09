"""BEDPE loops -> CTCF orientation probabilities (columns 8, 9) read from the reference genome.

Pipeline: read loops, keep those inside the region with enough PET counts, scan each distinct anchor once
for motif hits on both strands, convert the hits of an anchor into P(reverse), write an annotated BEDPE.
"""
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from loopsage.logger import get_logger
from loopsage.motif_finder.fasta import Fasta, norm_chrom
from loopsage.motif_finder.pssm import load_pfm

log = get_logger(__name__)
COLS = ["chr1", "start1", "end1", "chr2", "start2", "end2"]


@dataclass
class MotifResult:
    path: Path                                   # annotated BEDPE
    loops: pd.DataFrame                          # one row per kept loop (coordinates, pet, p1, p2, best scores)
    anchors: pd.DataFrame                        # one row per distinct anchor scanned
    stats: dict = field(default_factory=dict)    # counts at each filtering step, run parameters

    def save(self, folder) -> None:
        folder = Path(folder)
        self.loops.to_csv(folder / "motif_loops.tsv", sep="\t", index=False)
        self.anchors.to_csv(folder / "motif_anchors.tsv", sep="\t", index=False)
        (folder / "motif_stats.json").write_text(json.dumps(dict(self.stats, path=str(self.path))))

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        if not (folder / "motif_stats.json").is_file():
            return None
        st = json.loads((folder / "motif_stats.json").read_text())
        return cls(Path(st.pop("path")), pd.read_csv(folder / "motif_loops.tsv", sep="\t"),
                   pd.read_csv(folder / "motif_anchors.tsv", sep="\t"), st)


# ---- Read loops, filter by region and PET count ----------------------------

def read_loops(path) -> pd.DataFrame:
    """BEDPE as a table (first 7 columns named, extras kept); rows with bad coordinates are dropped."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, comment="#", dtype=str, engine="python")
    if raw.shape[1] < 6:
        raise ValueError(f"{path}: a BEDPE needs at least 6 columns, found {raw.shape[1]}")
    df = raw.copy()
    for c in (1, 2, 4, 5):
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    ok = df[[1, 2, 4, 5]].notna().all(axis=1)
    df = df[ok].copy()
    for c in (1, 2, 4, 5):
        df[c] = df[c].astype(np.int64)
    df[0], df[3] = df[0].map(norm_chrom), df[3].map(norm_chrom)
    df.attrs["malformed"] = int((~ok).sum())
    return df.reset_index(drop=True)


def select_loops(df, region, min_pet):
    """Loops with both anchors inside region (chrom, start, end) and pet >= min_pet; returns (table, counts)."""
    chrom, start, end = region
    inside = (df[0] == chrom) & (df[3] == chrom)
    for c in (1, 2, 4, 5):
        inside &= (df[c] >= start) & (df[c] < end)
    d = df[inside].copy()
    has_pet = d.shape[1] > 6 and pd.to_numeric(d[6], errors="coerce").notna().any()
    d["pet"] = pd.to_numeric(d[6], errors="coerce") if has_pet else np.nan
    n_region, pet_filtered = len(d), False
    if min_pet > 0:
        if has_pet:
            d = d[d["pet"].fillna(0) >= min_pet]
            pet_filtered = True
        else:
            log.warning("No PET / score column (7th) found: the minimum count filter is skipped.")
    d = d.drop_duplicates(subset=[0, 1, 2, 3, 4, 5]).reset_index(drop=True)   # identical loops count once
    return d, dict(n_region=int(n_region), n_kept=int(len(d)), pet_filtered=pet_filtered, has_pet=bool(has_pet))


# ---- Scan the anchors ------------------------------------------------------

def anchor_call(pssm, seq, cutoff):
    """Motif evidence for one anchor: n hits, best score, best strand, best position (0-1 within the anchor), P(reverse).

    P(reverse) = share of reverse hits when each hit is weighted by 2^score (its likelihood ratio); -1 = no motif.
    """
    start, strand, score = pssm.scan(seq, cutoff)
    if len(score) == 0:
        return 0, np.nan, 0, np.nan, -1.0
    w = 2.0 ** (score - score.max())
    b = int(score.argmax())
    centre = (start[b] + pssm.length / 2) / len(seq)
    return len(score), float(score[b]), int(strand[b]), float(centre), float(w[strand < 0].sum() / w.sum())


def scan_anchors(fasta, pssm, anchors, cutoff, progress, pad=0):
    """Table over distinct anchors (chrom, start, end) with one call each."""
    rows = []
    with progress("Searching CTCF motifs", len(anchors)) as advance:
        for chrom, s, e in anchors:
            seq = fasta.fetch(chrom, s - pad, e + pad) if fasta.has(chrom) else ""
            if len(seq) < pssm.length:
                rows.append((chrom, s, e, 0, np.nan, 0, np.nan, -1.0, 0.0, False))
            else:
                n, best, strand, pos, p = anchor_call(pssm, seq, cutoff)
                rows.append((chrom, s, e, n, best, strand, pos, p, seq.count("N") / len(seq), True))
            advance()
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "n_hits", "best_score", "best_strand", "best_pos", "p_rev", "frac_n", "scanned"])


def random_hit_rate(stats, anchors) -> float:
    """Chance that a random anchor of median length has at least one false hit (both strands)."""
    ln = np.median((anchors["end"] - anchors["start"]).to_numpy()) + 2 * stats.get("pad", 0)
    return float(1 - (1 - stats["pvalue"]) ** (2 * max(ln - stats["motif_length"] + 1, 0)))


# ---- Orchestration ---------------------------------------------------------

def annotate_bedpe(path, out_path, region, genome=None, pfm=None, pvalue=1e-5, pseudocount=0.8, min_pet=0, progress=None, pad=0):
    """Filter loops and (with a genome FASTA) add P(reverse) of both anchors as columns 8 and 9.

    region = (chrom, start, end) of the modelled window; only loops inside it are scanned.
    progress(desc, total) is a context manager yielding advance(n=1); default = none. pad = bp added on both sides of every anchor.
    """
    progress = progress or (lambda desc, total: nullcontext(lambda n=1: None))
    df = read_loops(path)
    loops, cnt = select_loops(df, region, min_pet)
    stats = dict(cnt, n_file=int(len(df)), malformed=df.attrs.get("malformed", 0), min_pet=int(min_pet), scanned=genome is not None,
                 pvalue=float(pvalue), pseudocount=float(pseudocount), pad=int(pad))
    log.step("BEDPE: %d loops read, %d inside %s:%d-%d, %d kept (min PET %d)", len(df), cnt["n_region"], *region, len(loops), min_pet)
    if loops.empty:
        raise ValueError(f"no loops left after filtering (region {region[0]}:{region[1]}-{region[2]}, min PET {min_pet})")

    out = loops[[0, 1, 2, 3, 4, 5]].copy()
    out.columns = COLS
    out["pet"] = loops["pet"].fillna(1.0).to_numpy() if cnt["has_pet"] else 1.0
    anchors = pd.DataFrame()
    if genome is None:
        for k, c in enumerate((7, 8)):                        # keep orientation columns the file already had
            if c in loops and pd.to_numeric(loops[c], errors="coerce").notna().all():
                out[f"p{k + 1}"] = pd.to_numeric(loops[c])
    else:
        pssm = load_pfm(pfm, pseudocount)
        cutoff = pssm.cutoff(pvalue)
        stats.update(motif=pssm.name, motif_length=pssm.length, cutoff=float(cutoff), max_score=pssm.max_score, consensus=pssm.consensus())
        fa = Fasta(genome)
        try:
            if not fa.has(region[0]):
                raise ValueError(f"{region[0]} not found in {genome} (contigs: {', '.join(list(fa.index)[:5])}...)")
            keys = sorted({(r.chr1, r.start1, r.end1) for r in out.itertuples()} | {(r.chr2, r.start2, r.end2) for r in out.itertuples()})
            log.step("Searching CTCF motifs in %d distinct anchors (%s, %d bp, score >= %.1f of max %.1f, p <= %g)",
                     len(keys), pssm.name, pssm.length, cutoff, pssm.max_score, pvalue)
            anchors = scan_anchors(fa, pssm, keys, cutoff, progress, pad)
        finally:
            fa.close()
        look = anchors.set_index(["chrom", "start", "end"])
        for k in (1, 2):
            idx = pd.MultiIndex.from_arrays([out[f"chr{k}"], out[f"start{k}"], out[f"end{k}"]])
            sub = look.reindex(idx)
            out[f"p{k}"] = sub["p_rev"].to_numpy()
            out[f"score{k}"] = sub["best_score"].to_numpy()
            out[f"n{k}"] = sub["n_hits"].fillna(0).astype(int).to_numpy()
        ok = anchors["scanned"]
        stats.update(n_anchors=int(len(anchors)), n_scanned=int(ok.sum()), n_with_motif=int((anchors["n_hits"] > 0).sum()),
                     n_both=int(((out["p1"] >= 0) & (out["p2"] >= 0)).sum()), n_one=int(((out["p1"] >= 0) ^ (out["p2"] >= 0)).sum()),
                     n_none=int(((out["p1"] < 0) & (out["p2"] < 0)).sum()), overwritten=bool(df.shape[1] > 8))
        log.step("Motifs found at %d/%d anchors with a hit; loops with both / one / no oriented anchor: %d / %d / %d",
                 stats["n_with_motif"], stats["n_anchors"], stats["n_both"], stats["n_one"], stats["n_none"])
        if stats["n_scanned"] < len(anchors):
            log.warning("%d anchors could not be read from the genome (missing contig or out of range)", len(anchors) - stats["n_scanned"])

    write = out[COLS + ["pet"] + [c for c in ("p1", "p2") if c in out]].copy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write.to_csv(out_path, sep="\t", header=False, index=False, float_format="%.4f")
    log.info("Annotated BEDPE written to %s", out_path)
    return MotifResult(Path(out_path), out, anchors, stats)
