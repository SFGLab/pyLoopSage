"""Parameter optimizer: loss, search and the --optimize pipeline stage."""
import json

import pytest
import numpy as np

from optimizer.nelder_mead import _clip, optimize
from analysis.maps import loop_map
from optimizer.objective import COMMUNICATION, loss_parts, loss_weights, weighted
from pipeline.runner import run_pipeline
from settings.loader import load_config


def _map(n=60, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return (m + m.T) / 2 + 3.0 / (1 + sep)


def test_loss_is_zero_for_identical_maps_and_positive_for_different_ones():
    m = _map()
    assert max(abs(v) for v in loss_parts(m, m)) < 1e-9
    assert sum(loss_parts(m, _map(seed=5))) > 0.1


def test_eigenvector_and_insulation_terms_are_normalised_distances():
    m = _map()
    p = loss_parts(m, _map(seed=5))
    assert 0 <= p[0] <= 1 and 0 <= p[2] <= 1
    assert loss_parts(m, 2.5 * m)[0] < 1e-6                         # only the shape counts, not the scale


def test_potts_state_replaces_the_eigenvector_term():
    m, ref = _map(), np.sin(np.arange(60) / 6.0)
    same, flipped = loss_parts(m, m, 5, ref, ref)[0], loss_parts(m, m, 5, -ref, ref)[0]
    assert same < 1e-9 and abs(flipped - 1) < 1e-9                 # (1 - r) / 2: B must be negative, a mirror image is the worst score
    assert abs(loss_parts(m, m, 5, np.ones(60), ref)[0] - 0.5) < 1e-9


def test_sparse_targets_get_smoothed_profiles():
    m, t = _map(), _map(seed=3)
    p = loss_parts(m, t, 5, smooth=(5.0, 2.0))
    assert np.isfinite(p).all() and 0 <= p[2] <= 1 and p != loss_parts(m, t, 5)


def test_weights_follow_the_region_size():
    tad, mid, comp = loss_weights(1e6), loss_weights(3e7), loss_weights(2e8)
    assert tad[2] > 3 * tad[0] and comp[0] > 3 * comp[2]            # insulation dominates TAD regions, eigenvectors compartment regions
    assert abs(mid[0] - mid[2]) < 0.05 and loss_weights(5e6)[0] < 0.02 < loss_weights(2e7)[0] < loss_weights(6e7)[0]   # PC1 ~ 0 for TADs, then grows
    assert weighted([0, 0, 0, 1], mid) > 0


def test_clip_respects_bounds():
    K, f, b, t = _clip([1000, 50.0, -50.0, 9.0], (3, 40))
    assert K == 40 and 0.05 - 1e-9 <= f <= 3 + 1e-9 and 0.25 - 1e-9 <= b <= 3 + 1e-9 and 1 - 1e-9 <= t <= 3 + 1e-9


def test_search_returns_the_best_checked_candidate():
    calls = []

    def obj(K, f, b, T=1.0, longer=1):
        calls.append(longer)
        obj.parts = [0.1, 0.2, 0.3]
        return (K - 20) ** 2 / 400 + (np.log(f) + 1) ** 2 + (np.log(b) + 1) ** 2 + (np.log(T) - 0.3) ** 2

    r = optimize(obj, 10, 1.0, 1.0, 300, max_evals=40)
    assert r.best["loss"] <= r.start["loss"] and r.evaluations <= 41 and 3 in calls      # long check ran
    assert abs(r.best["K"] - 20) <= 6 and 0.25 <= r.best["T"] <= 4 and len(r.best["parts"]) == 3


def test_early_stopping_ends_a_flat_search_and_can_be_switched_off():
    class Flat:                                                    # a loss that never improves
        parts, calls = (0.0, 0.0, 0.0, 0.0), 0

        def __call__(self, K, f, b, T=1.0, longer=1):
            return 2.0

    early = optimize(Flat(), 20, 1.0, 1.0, 200, max_evals=60, patience=10)
    assert early.stopped_early and not early.converged and early.evaluations < 20
    full = optimize(Flat(), 20, 1.0, 1.0, 200, max_evals=30, patience=0)
    assert not full.stopped_early and full.evaluations > early.evaluations


def test_optimize_flag_changes_the_production_parameters(tiny_args):
    ctx = run_pipeline(load_config(tiny_args + ["--optimize", "--opt-evals", "6", "--opt-steps", "600"]))
    b = ctx.optimization.best
    assert ctx.cfg.n_lef == b["K"] and ctx.cfg.folding_coeff == b["f"] and ctx.cfg.bind_coeff == b["b"]
    assert ctx.mc.meta["n_lef"] == b["K"] and ctx.cfg.n_lef2 == 0 and ctx.cfg.t_final == b["t_final"]
    saved = json.loads((ctx.dirs.data / "metadata" / "optimized_parameters.json").read_text())
    assert saved["chosen"]["n_lef"] == b["K"] and "verdict" in saved
    assert (ctx.dirs.figures / "optimizer_landscape.png").exists()
    assert (ctx.dirs.figures / "optimizer.png").exists()
    assert "Parameter optimizer" in (ctx.dirs.root / "report" / "report.html").read_text()
    assert "optimizer" in (ctx.dirs.config).read_text() and "What the optimizer explored" in (ctx.dirs.root / "report" / "report.html").read_text()


@pytest.mark.parametrize("mode", ["contrastive", "2D loop map", "preferential loading"])
def test_optimizer_works_with_every_binding_mode(tiny_args, tmp_path, mode):
    """Contrastive binding, the 2D loop map and preferential LEF loading all run through --optimize and never worsen the start."""
    extra = {"contrastive": ["--set", "contrastive_binding=true"], "2D loop map": ["--set", "data_loss_mode=1"]}.get(mode)
    if extra is None:
        pyBigWig = pytest.importorskip("pyBigWig")
        bw = str(tmp_path / "lef.bw")
        f = pyBigWig.open(bw, "w")
        f.addHeader([("chr1", 249250621)])
        f.addEntries("chr1", 77_000_000, values=[1 + np.sin(i / 5) for i in range(1800)], span=10_000, step=10_000)
        f.close()
        extra = ["--set", f"lef_track_file={bw}"]
    ctx = run_pipeline(load_config(tiny_args + ["--optimize", "--opt-evals", "6", "--opt-steps", "500"] + extra))
    r = ctx.optimization
    assert len(r.best["parts"]) == 4 and np.isfinite(r.best["loss"]) and r.best["loss"] <= r.start["loss"] + 1e-9
    assert ctx.mc.meta["n_lef"] == r.best["K"]


def test_reported_map_is_the_one_the_optimizer_scored(tiny_args):
    ctx = run_pipeline(load_config(tiny_args + ["--optimize", "--opt-evals", "5", "--opt-steps", "600"]))
    same = loop_map(ctx.mc_ens, ctx.geom.n_beads, **COMMUNICATION)
    assert np.allclose(ctx.mc_map, same)                          # insulation and validation use the optimized map definition
    plain = run_pipeline(load_config(tiny_args))
    assert np.allclose(plain.mc_map, loop_map(plain.mc_ens, plain.geom.n_beads))      # default runs are unchanged


def test_loop_coverage_and_decay_count_the_whole_area_between_anchors():
    from analysis.maps import loop_coverage, loop_decay
    c = np.zeros((10, 10)); c[2, 6] = c[6, 2] = 2.0
    cov, dec = loop_coverage(c), loop_decay(c)
    assert list(cov) == [0, 0, 2, 2, 2, 2, 0, 0, 0, 0]
    assert np.allclose(dec[1:6] * (10 - np.arange(1, 6)), [6, 4, 2, 0, 0])      # loop of length 4: 2 x (4 - s) pairs at separation s
    m, t = _map(seed=1), _map(seed=5)
    z = np.zeros(len(m) - 10)
    prof = (np.r_[cov, z], np.r_[dec, z] + 1e-3)
    assert loss_parts(m, t, 5, cover=(prof, prof))[2] < 1e-9


def test_potts_search_stays_in_bounds_and_finds_the_optimum():
    from optimizer.nelder_mead import optimize_potts

    def obj(e, p, longer=1):
        obj.parts = [0.1]
        return (np.log(e) - np.log(2.0)) ** 2 + (p - 0.3) ** 2

    r = optimize_potts(obj, 1.0, 0.5, max_evals=40)
    lo, hi = r["bounds"]["epi"]
    assert lo <= r["best"]["epi"] <= hi and 0.1 <= r["best"]["p"] <= 0.9
    assert r["best"]["loss"] <= r["start"]["loss"] and abs(r["best"]["epi"] - 2.0) < 1.0


def test_loops_only_stage_skips_the_eigenvector_term():
    m, t = _map(), _map(seed=5)
    assert loss_parts(m, t, 5, eig=False)[0] == 0.0 and loss_parts(m, t, 5)[0] > 0


def test_potts_run_fits_the_loop_stage_first_and_then_the_potts_stage(tiny_args, tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    bw = str(tmp_path / "comp.bw")
    f = pyBigWig.open(bw, "w")
    f.addHeader([("chr1", 249250621)])
    f.addEntries("chr1", 77_000_000, values=[float(np.sign(np.sin(i / 20))) for i in range(1800)], span=10_000, step=10_000)
    f.close()
    ctx = run_pipeline(load_config(tiny_args + ["--optimize", "--opt-evals", "6", "--opt-steps", "500", "--set", "epi_coeff=1.0", "--set", f"comp_bw_file={bw}"]))
    r = ctx.optimization
    assert r.potts and r.best["parts"][0] == 0.0                   # stage 1 has no eigenvector term; stage 2 has it, weighted
    assert 0.1 <= r.potts["best"]["p"] <= 0.9 and r.potts["best"]["loss"] <= r.potts["start"]["loss"] + 1e-9
    assert ctx.cfg.epi_coeff == r.potts["best"]["epi"]
