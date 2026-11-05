"""Parameter optimizer: loss, search and the --optimize pipeline stage."""
import json

import pytest
import numpy as np

from optimizer.nelder_mead import _clip, optimize
from optimizer.objective import loss_parts, WEIGHTS, weighted
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


def test_insulation_weighs_more_than_the_decay_and_short_loops_cost():
    assert WEIGHTS[2] > WEIGHTS[1] and weighted([0, 1, 0]) < weighted([0, 0, 1]) and weighted([0, 0, 0, 1]) > 0


def test_clip_respects_bounds():
    K, f, b, t = _clip([1000, 50.0, -50.0, 9.0], (3, 40))
    assert K == 40 and 0.01 <= f <= 3 and 0.005 <= b <= 3 and 0.25 <= t <= 4


def test_search_returns_the_best_checked_candidate():
    calls = []

    def obj(K, f, b, T=1.0, longer=1):
        calls.append(longer)
        obj.parts = [0.1, 0.2, 0.3]
        return (K - 20) ** 2 / 400 + (np.log(f) + 1) ** 2 + (np.log(b) + 1) ** 2 + (np.log(T) - 0.3) ** 2

    r = optimize(obj, 10, 1.0, 1.0, 300, max_evals=40)
    assert r.best["loss"] <= r.start["loss"] and r.evaluations <= 41 and 3 in calls      # long check ran
    assert abs(r.best["K"] - 20) <= 6 and 0.25 <= r.best["T"] <= 4 and len(r.best["parts"]) == 3


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
