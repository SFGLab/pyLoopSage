"""CTCF motif finder: PSSM scan, indexed FASTA, orientation calls, loop filtering, config flags and the report."""
import re

import numpy as np
import pandas as pd
import pytest

from loopsage.motif_finder import annotate_bedpe, load_pfm
from loopsage.motif_finder.annotate import anchor_call, read_loops, select_loops
from loopsage.motif_finder.fasta import encode, Fasta
from loopsage.pipeline.runner import run_pipeline
from loopsage.settings.loader import load_config

COMP = str.maketrans("ACGT", "TGCA")
REGION = ("chr1", 1_000_000, 1_600_000)


def revcomp(s):
    return s.translate(COMP)[::-1]


# ---- Synthetic genome with planted motifs ----------------------------------

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """chr1 (1.7 Mb random) + loops with planted CTCF sites of known orientation."""
    d = tmp_path_factory.mktemp("motif")
    rng = np.random.default_rng(7)
    seq = rng.choice(list("ACGT"), 1_700_000)
    cons = load_pfm().consensus()
    rows, truth = [], []
    starts = np.sort(rng.choice(np.arange(1_010_000, 1_590_000, 6000), 40, replace=False))
    kinds = ["conv"] * 14 + ["div"] * 6 + ["tandem"] * 6 + ["empty"] * 4 + ["conv"] * 10
    for i in range(0, 40, 2):
        a, b, kind = starts[i], starts[i + 1], kinds[i // 2]
        for pos, strand in ((a, "+"), (b, "-")):
            site = {"conv": cons if pos == a else revcomp(cons), "div": revcomp(cons) if pos == a else cons,
                    "tandem": cons, "empty": None}[kind]
            if site:
                seq[pos + 1500:pos + 1500 + len(site)] = list(site)
        pet = 1 + (i // 2) % 8
        rows.append(f"chr1\t{a}\t{a + 3000}\tchr1\t{b}\t{b + 3000}\t{pet}")
        truth.append(kind)
    seq[:2000] = "N"                                       # an N stretch at the start
    fa = d / "genome.fa"
    s = "".join(seq)
    fa.write_text(">chr1 test\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60)) + "\n>chr2\nACGT\n")
    bedpe = d / "loops.bedpe"
    bedpe.write_text("# header\n" + "\n".join(rows + ["chr1\t5\t100\tchr1\t900\t990\t9", "chr1\tbad\t1\tchr1\t1\t2\t3"]) + "\n")
    return dict(dir=d, fasta=fa, bedpe=bedpe, kinds=truth, seq=s)


# ---- PSSM and FASTA --------------------------------------------------------

def test_pssm_matches_biopython():
    motifs = pytest.importorskip("Bio.motifs")
    from Bio.Seq import Seq
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 3000))
    cons = load_pfm().consensus()
    seq = seq[:500] + cons + seq[519:2000] + revcomp(cons) + seq[2019:]
    with open(load_pfm.__globals__["DEFAULT_PFM"]) as f:
        pssm = motifs.read(f, "pfm").counts.normalize(pseudocounts=0.2).log_odds()
    mine = load_pfm(pseudocount=0.8)
    cut = 6.0
    ref = {(int(p) if p >= 0 else len(seq) + int(p), 1 if p >= 0 else -1): float(s) for p, s in pssm.search(seq, threshold=cut)}
    s, st, sc = mine.scan(seq, cut)
    got = {(int(a), int(b)): float(c) for a, b, c in zip(s, st, sc)}
    assert set(got) == set(ref)
    assert all(abs(got[k] - ref[k]) < 1e-3 for k in ref)
    assert (500, 1) in got and (2000, -1) in got


def test_cutoff_is_calibrated():
    p = load_pfm()
    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list("ACGT"), 300_000))
    c = p.cutoff(1e-4)
    n = len(p.scan(seq, c)[2])
    assert 0.6 * 60 < n < 1.5 * 60                            # expected 2 strands * 300k * 1e-4 = 60
    assert p.cutoff(1e-6) > p.cutoff(1e-4) > p.cutoff(1e-2)


def test_n_never_scores_a_hit():
    p = load_pfm()
    assert len(p.scan("N" * 100, -50.0)[2]) == 0
    assert (encode("ACGTNacgt") == [0, 1, 2, 3, 4, 0, 1, 2, 3]).all()


def test_fasta_fetch_index_and_aliases(world):
    fa = Fasta(world["fasta"])
    s = world["seq"]
    assert fa.fetch("chr1", 59, 61) == s[59:61]                  # across a line break
    assert fa.fetch("1", 1_000_000, 1_000_500) == s[1_000_000:1_000_500]
    assert fa.fetch("chr1", 1_699_990, 2_000_000) == s[1_699_990:]  # clipped at the contig end
    assert fa.fetch("chr1", 5, 5) == "" and fa.has("chr2") and not fa.has("chr9")
    assert fa.length("chr2") == 4
    assert (world["dir"] / "genome.fa.fai").exists()
    fa.close()
    Fasta(world["fasta"]).close()                               # second open reuses the .fai


def test_anchor_call_weights():
    p = load_pfm()
    cons = p.consensus()
    flank = "ACGT" * 20
    c = p.cutoff(1e-5)
    assert anchor_call(p, flank, c)[0] == 0 and anchor_call(p, flank, c)[4] == -1.0
    n, best, strand, pos, prob = anchor_call(p, flank + cons + flank, c)
    assert n >= 1 and strand == 1 and prob < 0.01 and 0.3 < pos < 0.7
    assert anchor_call(p, flank + revcomp(cons) + flank, c)[4] > 0.99
    mixed = anchor_call(p, flank + cons + flank + revcomp(cons[:3] + "A" + cons[4:]) + flank, c)[4]    # weaker reverse site
    assert 0 < mixed < 0.5


# ---- Loops: reading, filtering, annotation ---------------------------------

def test_read_and_select(world):
    df = read_loops(world["bedpe"])
    assert df.attrs["malformed"] == 1 and len(df) == 21
    d, cnt = select_loops(df, REGION, min_pet=0)
    assert cnt["n_region"] == 20                               # the loop at 5-990 is outside
    d, cnt = select_loops(df, REGION, min_pet=5)
    assert cnt["pet_filtered"] and (d["pet"] >= 5).all() and cnt["n_kept"] == len(d) < 20


def test_orientation_recovery(world, tmp_path):
    res = annotate_bedpe(world["bedpe"], tmp_path / "out.bedpe", REGION, world["fasta"])
    lo = res.loops
    assert len(lo) == 20 and res.stats["n_scanned"] == res.stats["n_anchors"] == 40
    ok = {"conv": lambda r: r.p1 < 0.05 and r.p2 > 0.95, "div": lambda r: r.p1 > 0.95 and 0 <= r.p2 < 0.05,
          "tandem": lambda r: r.p1 < 0.05 and r.p2 < 0.05, "empty": lambda r: r.p1 < 0 and r.p2 < 0}
    hits = [ok[k](r) for k, r in zip(world["kinds"], lo.itertuples())]
    assert np.mean(hits) >= 0.9                                # a rare random false hit may flip an 'empty' anchor
    assert res.stats["n_both"] >= 14 and res.stats["n_none"] <= 6
    best = res.anchors["best_pos"].dropna()
    assert ((best > 0.4) & (best < 0.6)).mean() > 0.9           # motif planted in the middle of the anchor


def test_output_file_is_valid_loopsage_input(world, tmp_path):
    from loopsage.analysis.inputs import anchor_table
    from loopsage.common.genome import Geometry
    res = annotate_bedpe(world["bedpe"], tmp_path / "out.bedpe", REGION, world["fasta"], min_pet=3)
    cols = pd.read_csv(res.path, sep="\t", header=None)
    assert cols.shape[1] == 9 and (cols[6] >= 3).all()
    g = Geometry("chr1", 1_000_000, 1_600_000, 100)
    a = anchor_table(res.path, g)
    assert a is not None and (a.p_rev >= -1).all() and (a.config == "convergent").sum() >= 5


def test_filter_only_without_genome(world, tmp_path):
    res = annotate_bedpe(world["bedpe"], tmp_path / "f.bedpe", REGION, None, min_pet=6)
    assert not res.stats["scanned"] and res.anchors.empty
    assert pd.read_csv(res.path, sep="\t", header=None).shape[1] == 7


def test_errors(world, tmp_path):
    with pytest.raises(ValueError, match="no loops left"):
        annotate_bedpe(world["bedpe"], tmp_path / "x.bedpe", REGION, None, min_pet=999)
    with pytest.raises(FileNotFoundError):
        Fasta(world["dir"] / "none.fa")
    other = tmp_path / "chr2only.fa"
    other.write_text(">chr2\nACGTACGT\n")
    with pytest.raises(ValueError, match="not found in"):                # contig absent from the genome
        annotate_bedpe(world["bedpe"], tmp_path / "x.bedpe", REGION, other)


# ---- Config and report -----------------------------------------------------

def test_cli_flags_and_validation(world, tmp_path):
    base = ["-f", str(world["bedpe"]), "-r", "chr1:1000000-1600000", "-o", str(tmp_path)]
    cfg = load_config(base + ["--motif-genome", str(world["fasta"]), "--min-pet", "3", "--motif-pvalue", "1e-4"])
    assert cfg.min_pet == 3 and cfg.motif_genome == str(world["fasta"]) and cfg.motif_pvalue == 1e-4 and cfg.motif_pfm is None
    d = load_config(base)
    assert d.min_pet == 0 and d.motif_genome is None                # off by default
    for bad in (["--motif-genome", str(tmp_path / "missing.fa")], ["--motif-pvalue", "2"], ["--min-pet", "-1"]):
        with pytest.raises(SystemExit):                           # CLI errors exit with a message
            load_config(base + bad)
    bed = tmp_path / "x.bed"
    bed.write_text("chr1\t1\t2\n")
    with pytest.raises(SystemExit):
        load_config(["-f", str(bed), "-r", "chr1:1000000-1600000", "--min-pet", "2"])


def test_pipeline_report_has_motif_diagnostics(world, tmp_path):
    cfg = load_config(["-f", str(world["bedpe"]), "-r", "chr1:1000000-1600000", "-n", "200", "-o", str(tmp_path / "res"),
                       "--motif-genome", str(world["fasta"]), "--min-pet", "2", "--mc-steps", "1500", "--sample-every", "100",
                       "--burnin", "300", "--no-run-MD", "--seed", "1"])
    ctx = run_pipeline(cfg)
    root = ctx.dirs.root
    html = (root / "report" / "report.html").read_text()
    assert "CTCF motif search" in html and "Anchors with a motif" in html
    assert (root / "visualizations" / "motif_search.png").exists()
    assert not (root / "visualizations" / "motif_loops.png").exists()      # one compact figure only
    assert (root / "data" / "loops_annotated.bedpe").exists() and (root / "data" / "motif_stats.json").exists()
    assert ctx.motif.stats["n_both"] > 5 and ctx.inputs.loops.stats["n_loops"] > 5
    assert re.search(r"Orientation", html)                      # the usual anchor tables now use the motif calls


# ---- Plain BEDPE import still works (motif search is optional) -------------

def _variant(src, dst, ncols):
    rows = [line.split("\t")[:ncols] for line in src.read_text().splitlines() if line.strip()]
    dst.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return dst


@pytest.mark.parametrize("ncols", [6, 7, 9])
def test_plain_bedpe_loads_with_or_without_orientation(demo_bedpe, tmp_path, ncols):
    from loopsage.common.genome import Geometry
    from loopsage.preproc.loader import load_inputs
    f = _variant(demo_bedpe, tmp_path / f"l{ncols}.bedpe", ncols)
    cfg = load_config(["-f", str(f), "-r", "chr1:77000000-95000000", "-n", "120", "-o", str(tmp_path)])
    assert cfg.min_pet == 0 and cfg.motif_genome is None                       # nothing extra runs by default
    lo = load_inputs(cfg, Geometry("chr1", 77_000_000, 95_000_000, 120)).loops
    assert lo.stats["n_loops"] > 10
    if ncols == 9:
        assert not np.allclose(lo.L, lo.R)                                      # orientation columns are used
    else:
        assert np.allclose(lo.L, lo.R)                                          # no orientation: split evenly


def test_default_run_has_no_motif_step(demo_bedpe, tmp_path):
    f = _variant(demo_bedpe, tmp_path / "l7.bedpe", 7)
    cfg = load_config(["-f", str(f), "-r", "chr1:77000000-95000000", "-n", "120", "-o", str(tmp_path / "r"), "--mc-steps", "1500",
                       "--sample-every", "100", "--burnin", "300", "--no-run-MD", "--seed", "1", "--no-figures"])
    ctx = run_pipeline(cfg)
    assert ctx.motif is None and not (ctx.dirs.data / "loops_annotated.bedpe").exists()
    assert ctx.loops_file == str(f)
    assert "Anchors with a motif" not in (ctx.dirs.root / "report" / "report.html").read_text()
    assert not (ctx.dirs.figures / "motif_search.png").exists()


def test_pet_filter_keeps_existing_orientation_columns(demo_bedpe, tmp_path):
    res = annotate_bedpe(demo_bedpe, tmp_path / "o.bedpe", ("chr1", 77_000_000, 95_000_000), None, min_pet=20)
    out = pd.read_csv(res.path, sep="\t", header=None)
    src = pd.read_csv(demo_bedpe, sep="\t", header=None)
    assert out.shape[1] == 9 and (out[6] >= 20).all() and len(out) == (src[6] >= 20).sum()
    merged = out.merge(src, on=[0, 1, 2, 3, 4, 5])
    assert np.allclose(merged["7_x"], merged["7_y"], atol=1e-3)                  # probabilities unchanged
