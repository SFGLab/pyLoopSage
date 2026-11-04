"""Optional Hi-C reference: .hic reader, mask-aware loss, validation scores, report figures and the optimizer target."""
import json

import numpy as np
import pytest

from analysis.maps import fill_masked
from common.genome import Geometry
from optimizer.objective import loss_parts, target_map
from pipeline.runner import run_pipeline
from preproc import hic as H
from settings.loader import load_config
from validation.hic_compare import compare_hic


def _hic_like(n=120, seed=0, noise=0.1):
    """Synthetic Hi-C: power-law decay, two domains and noise."""
    rng = np.random.default_rng(seed)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    m = 100.0 / (1 + sep) ** 1.0
    dom = (np.arange(n)[:, None] < n // 2) == (np.arange(n)[None, :] < n // 2)
    m = m * np.where(dom, 1.5, 0.7) * np.exp(rng.normal(0, noise, (n, n)))
    return (m + m.T) / 2


def test_resolution_choice_prefers_the_bead_size_and_skips_huge_matrices():
    assert H.candidate_resolutions([5000, 10000, 25000, 1000], 10000, 2_000_000)[0] == 10000
    assert H.candidate_resolutions([5000, 10000], 10000, 2_000_000, wanted=5000)[0] == 5000
    assert 1000 not in H.candidate_resolutions([1000, 10000], 10000, 100_000_000)        # > MAX_BINS bins


def test_rebin_conserves_counts_for_finer_bins_and_marks_uncovered_beads():
    g = Geometry("chr1", 0, 400_000, 20)                                                 # 20 kb beads
    fine = np.ones((40, 40))                                                             # 10 kb bins
    out, cover = H.rebin(fine, 10_000, 0, g)
    assert out.shape == (20, 20) and np.allclose(out, 4.0) and cover.min() == 1.0
    coarse, cov = H.rebin(np.ones((4, 4)), 100_000, 0, Geometry("chr1", 0, 400_000, 20))
    assert coarse.shape == (20, 20) and np.isfinite(coarse).all() and np.allclose(coarse, 1 / 25)
    part, cov = H.rebin(np.ones((2, 2)), 100_000, 0, Geometry("chr1", 0, 400_000, 20))   # covers half the region
    assert np.isnan(part[-1]).all() and cov[-1] == 0


def test_masked_beads_are_neutral_in_the_loss():
    t = _hic_like()
    m = _hic_like(seed=3)
    masked = t.copy()
    masked[10:14, :] = masked[:, 10:14] = np.nan
    assert np.isfinite(fill_masked(masked)).all()
    parts = loss_parts(m, masked)
    assert np.isfinite(parts).all() and abs(sum(parts) - sum(loss_parts(m, t))) < 0.3
    assert target_map(None, None, None, None, hic=masked).shape == t.shape


def test_hic_scores_are_high_for_the_same_map_and_lower_for_a_different_one():
    t = _hic_like()
    same, other = compare_hic(t * 3.0, t), compare_hic(_hic_like(seed=9, noise=1.5)[::-1, ::-1] ** 0.3, t)
    assert same["r_oe"] > 0.99 and same["r_ps"] > 0.99 and same["scc"] > 0.99
    assert other["r_oe"] < same["r_oe"] and other["r_log"] < same["r_log"]


def test_missing_hicstraw_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(H, "have_hicstraw", lambda: False)
    with pytest.raises(ImportError, match="hic-straw"):
        H.load_hic_region("x.hic", Geometry("chr1", 0, 100_000, 10))


def test_reads_a_real_hic_file(tmp_path):
    pytest.importorskip("hicstraw")
    from hic_writer import write_hic
    n, res = 60, 10_000
    mat = np.round(_hic_like(n) * 10).astype(int)
    path = write_hic(tmp_path / "t.hic", mat, "chr1", res)
    out, info = H.load_hic_region(str(path), Geometry("chr1", 0, n * res, n), norm="KR")
    assert out.shape == (n, n) and info["resolution"] == res and info["norm"] == "NONE"   # KR is absent: falls back
    assert np.corrcoef(np.triu(out, 1).ravel(), np.triu(mat, 1).ravel())[0, 1] > 0.99


@pytest.fixture
def fake_hic(monkeypatch, tmp_path):
    """Replace the .hic reader by a synthetic map on the run's bead grid."""
    def fake(path, geom, norm="KR", resolution=None, **kw):
        m = _hic_like(geom.n_beads)
        m[5:8, :] = m[:, 5:8] = np.nan                                                    # a masked stretch
        return m, dict(resolution=geom.bp_per_bead, norm="KR", chrom=geom.chrom, coverage=0.95, file="fake.hic")
    monkeypatch.setattr(H, "load_hic_region", fake)
    return str(tmp_path / "fake.hic")


def test_hic_in_the_report_validation_and_metadata(tiny_args, fake_hic):
    ctx = run_pipeline(load_config(tiny_args + ["--hic", fake_hic]))
    figs = ctx.dirs.figures
    assert all((figs / f"{k}.png").exists() for k in ("hic_input", "hic_validation", "hic_scores"))
    html = (ctx.dirs.root / "report" / "report.html").read_text()
    assert "Hi-C reference" in html and "Validation vs Hi-C" in html and "Simulation against Hi-C" in html
    meta = ctx.dirs.data / "metadata"
    assert (meta / "hic_map.npy").exists() and json.loads((meta / "hic_info.json").read_text())["file"] == "fake.hic"


def test_no_hic_means_no_hic_output(tiny_args):
    ctx = run_pipeline(load_config(tiny_args))
    assert ctx.hic is None and not (ctx.dirs.figures / "hic_validation.png").exists()


def test_optimizer_uses_the_hic_map_as_target(tiny_args, fake_hic):
    ctx = run_pipeline(load_config(tiny_args + ["--hic", fake_hic, "--optimize", "--opt-evals", "5", "--opt-steps", "500"]))
    html = (ctx.dirs.root / "report" / "report.html").read_text()
    assert ctx.optimization is not None and "Hi-C map (decay, insulation and O/E taken from it)" in html
