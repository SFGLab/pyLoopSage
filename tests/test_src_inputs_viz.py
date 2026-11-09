"""Anchor orientation tables, compartment domains and the input figures."""
from pathlib import Path

import numpy as np
import pytest

from loopsage.analysis.inputs import CONF_MIN, anchor_table, config_summary, orientation_summary
from loopsage.analysis.inputs import compartment_summary, domains
from loopsage.common.genome import Geometry

GEOM = Geometry("chr1", 0, 1_000_000, 100)


@pytest.fixture
def bedpe(tmp_path):
    """4 loops: convergent, tandem, divergent, uncertain (unknown orientation)."""
    rows = [(100_000, 400_000, 0.05, 0.95), (200_000, 500_000, 0.05, 0.1),
            (300_000, 600_000, 0.95, 0.05), (400_000, 800_000, -1, -1)]
    f = tmp_path / "t.bedpe"
    f.write_text("\n".join(f"chr1\t{a}\t{a + 5000}\tchr1\t{b}\t{b + 5000}\t{10 + i}\t{p1}\t{p2}"
                           for i, (a, b, p1, p2) in enumerate(rows)))
    return f


def test_loop_configurations(bedpe):
    a = anchor_table(bedpe, GEOM)
    assert list(a.config) == ["convergent", "tandem", "divergent", "uncertain"]
    assert dict((r[0], r[1]) for r in config_summary(a)) == dict(convergent=1, tandem=1, divergent=1, uncertain=1)


def test_orientation_and_confidence(bedpe):
    a = anchor_table(bedpe, GEOM)
    assert (a.orient == "unknown").sum() == 2 and (a.orient == "forward").sum() == 4
    assert a.confidence[a.p_rev == 0.5].size == 0 and a.confidence.max() == pytest.approx(0.9)
    assert sum(r[1] for r in orientation_summary(a)) == len(a.bp)


def test_anchor_order_is_left_to_right(tmp_path):
    f = tmp_path / "s.bedpe"
    f.write_text("chr1\t600000\t605000\tchr1\t100000\t105000\t5\t0.9\t0.1\n")      # anchors given right-first
    a = anchor_table(f, GEOM)
    assert a.left_bp[0] < a.right_bp[0] and a.config[0] == "convergent"


def test_missing_orientation_columns(tmp_path):
    f = tmp_path / "n.bedpe"
    f.write_text("chr1\t100000\t105000\tchr1\t400000\t405000\n")
    a = anchor_table(f, GEOM)
    assert a.config[0] == "uncertain" and (a.orient == "unknown").all()


def test_compartment_domains_merge_flickers():
    comp = np.r_[np.ones(10), -np.ones(1), np.ones(10), -np.ones(15)]
    assert domains(comp) == [(0, 20, "A"), (21, 35, "B")]


def test_compartment_summary_anchor_enrichment():
    comp = np.r_[np.ones(50), -np.ones(50)]
    anchors = np.r_[np.full(50, 2.0), np.full(50, 0.5)] + np.random.default_rng(0).uniform(0, 0.1, 100)
    s = compartment_summary(comp, anchors, GEOM)
    assert s["frac_a"] == 0.5 and s["n_domains"] == 2 and s["p"] < 0.01
    assert s["anchors_a"].mean() > s["anchors_b"].mean()


def test_demo_inputs_produce_all_input_figures(tmp_path, demo_bedpe):
    """Full demo-style run with BED compartments: every input figure exists and has a caption."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
    from make_demo_data import write_compartments
    from loopsage.pipeline.runner import run_pipeline
    from loopsage.settings.loader import load_config
    comp = write_compartments(tmp_path / "c.bed")
    ctx = run_pipeline(load_config(["-f", str(demo_bedpe), "-r", "chr1:77000000-95000000", "-n", "100", "-o", str(tmp_path / "r"),
                                    "--mc-steps", "1500", "--sample-every", "100", "--burnin", "300", "--no-run-MD",
                                    "--set", f"comp_bed_file={comp}"]))
    names = {p.stem for p in ctx.dirs.figures.glob("*.png")}
    assert {"anchor_map", "anchor_stats", "anchor_loops", "compartments", "lef_trajectories", "kymograph"} <= names
    html = ctx.dirs.report.read_text()
    assert "Compartments (A/B)" in html and "Input anchors: strength and CTCF orientation" in html
