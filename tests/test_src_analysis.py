"""Diagnostics, insight metrics, validation and 3D geometry on a tiny real run."""
import numpy as np
import pytest

from loopsage.analysis import ensemble
from loopsage.analysis.ensemble import acceptance_table, loop_lengths, occupancy
from loopsage.analysis.ensemble import acf, equilibrium_summary, integrated_time
from loopsage.analysis.maps import loop_map
from loopsage.monte_carlo.simulation import MCResult
from loopsage.monte_carlo.simulation import run_mc
from loopsage.preproc.loader import load_inputs
from loopsage.preproc.loader import build_geometry
from loopsage.settings.loader import load_config


@pytest.fixture(scope="module")
def run(tmp_path_factory, demo_bedpe):
    out = tmp_path_factory.mktemp("an")
    cfg = load_config(["-f", str(demo_bedpe), "-r", "chr1:77000000-95000000", "-n", "120", "-o", str(out),
                       "--mc-steps", "3000", "--sample-every", "100", "--burnin", "500", "--no-run-MD", "--seed", "3"])
    geom = build_geometry(cfg)
    inputs = load_inputs(cfg, geom)
    return cfg, geom, inputs, run_mc(cfg, inputs, geom)


def test_acf_properties():
    x = np.random.default_rng(0).normal(size=500)
    a = acf(x)
    assert a[0] == pytest.approx(1.0) and abs(a[5]) < 0.2
    assert integrated_time(x) < 3


def test_mc_result_roundtrip(run, tmp_path):
    *_, mc = run
    mc.save(tmp_path / "mc.npz")
    back = MCResult.load(tmp_path / "mc.npz")
    assert np.array_equal(back.Ms, mc.Ms) and back.meta["n_lef"] == mc.meta["n_lef"]


def test_loop_and_occupancy_stats(run):
    _, geom, _, mc = run
    assert (loop_lengths(mc) >= 0).all()
    occ = occupancy(mc, geom.n_beads)
    assert occ.shape == (geom.n_beads,) and occ.sum() == pytest.approx(2 * mc.Ms.shape[0])
    assert all(0 <= r[3] <= 1 for r in acceptance_table(mc))


def test_equilibrium_summary_keys(run):
    eq = equilibrium_summary(run[3])
    assert {"Es", "Fs", "tau", "ess"} <= set(eq) and eq["ess"] > 0


def test_loop_map_symmetric_nonnegative(run):
    mat = loop_map(run[3], run[1].n_beads)
    assert np.allclose(mat, mat.T) and mat.min() >= 0 and mat.sum() > 0


def test_insight_metrics_in_range(run):
    _, geom, inputs, mc = run
    rec, dyn = ensemble.anchor_recovery(mc, inputs), ensemble.lef_dynamics(mc)
    assert 0 <= rec["recall"] <= 1 and 0 <= rec["precision"] <= 1
    assert 0 <= dyn["stable_fraction"] <= 1 and dyn["dwell_steps"] > 0
    inp, sim, ks, _ = ensemble.length_agreement(mc, inputs, geom)
    assert 0 <= ks <= 1 and len(inp) > 0 and len(sim) > 0


def test_validation_runs_on_mc_map(run):
    from loopsage.validation.compare import validate
    cfg, geom, _, mc = run
    res = validate(loop_map(mc, geom.n_beads), cfg, geom)
    assert res.rows and all(-1 <= r["pearson"] <= 1 for r in res.rows)


def test_md_geometry_helpers():
    from loopsage.analysis.structure import distance_scaling
    rng = np.random.default_rng(1)
    coords = np.cumsum(rng.normal(size=(5, 200, 3)), axis=1).astype(np.float32)   # ideal random walks
    s, r, nu = distance_scaling(coords)
    assert 0.35 < nu < 0.65 and len(s) == len(r)


# ---- Loop map: dots, domains, communication and the epigenetic checkerboard ----

class _Fake:
    """Minimal MC result: one snapshot with given loops."""
    def __init__(self, loops, epi=None):
        self.Ms, self.Ns = (np.array([[l[k]] for l in loops]) for k in (0, 1))
        self.epi = epi

    def production(self, a):
        return a


def test_loop_map_dot_domain_and_symmetry():
    from loopsage.analysis.maps import loop_map
    m = loop_map(_Fake([(10, 30)]), 40, sigma=0.8)
    assert np.allclose(m, m.T)
    assert m[10, 30] > m[20, 25] > m[5, 35]            # dot > interior > outside
    assert m[10, 30] > 3 * m[10, 24]                   # dot is sharp


def test_loop_map_communication_is_weaker_and_recursive():
    from loopsage.analysis.maps import loop_map
    kw = dict(domain=0, base=0, sigma=0.3)
    chain = loop_map(_Fake([(5, 15), (15, 25), (25, 35)]), 40, **kw)
    one = loop_map(_Fake([(5, 15)]), 40, **kw)
    assert chain[5, 25] > 0 and chain[5, 25] < chain[5, 15]            # (i,k) from (i,j)+(j,k), weaker
    assert chain[5, 35] > 0 and chain[5, 35] < chain[5, 25]            # second hop weaker still
    assert one[5, 25] == pytest.approx(0, abs=1e-6)


def test_loop_map_communication_through_adjacent_anchors():
    from loopsage.analysis.maps import loop_map
    kw = dict(domain=0, base=0, sigma=0.3)
    adj = loop_map(_Fake([(5, 15), (16, 25)]), 40, **kw)             # anchors on beads 15 and 16
    far = loop_map(_Fake([(5, 15), (18, 25)]), 40, **kw)
    assert adj[5, 25] > 0 and far[5, 25] == pytest.approx(0, abs=1e-6)


def test_checkerboard_and_combination():
    from loopsage.analysis.maps import checkerboard, combine_maps, epigenetic_profile
    epi = np.array([[0, 0], [0, 0], [2, 2], [2, 2]])    # two B-like beads then two A-like beads, 3 states
    prof = epigenetic_profile(_Fake([(0, 1)], epi), 3)
    assert np.allclose(prof, [-1, -1, 1, 1])
    board = checkerboard(prof)
    assert board[0, 1] == 1 and board[0, 3] == -1
    comb = combine_maps(np.ones((4, 4)), board)
    assert comb[0, 1] > 1 > comb[0, 3] > 0


def test_lef_trajectory_segments_split_at_rebinding_jumps():
    from loopsage.viz.mc_figs import _segments
    m = np.array([10, 11, 12, 13, 80, 81, 82, 83, 84, 30, 31], float)
    n = m + 5
    assert _segments(m, n, jump=20) == [(0, 4), (4, 9)]                 # jumps at 4 and 9; the 2-snapshot stub at the end is dropped
    assert _segments(m[:4], n[:4], jump=20) == [(0, 4)]                 # no jump: one segment


# ---- Validation: smoothed correlations ----

def test_smoothing_helper():
    from loopsage.validation.compare import smooth
    v = np.zeros(50); v[25] = 10.0
    s = smooth(v, 2.0)
    assert s.max() < 10 and s.sum() == pytest.approx(v.sum(), rel=1e-6) and np.array_equal(smooth(v, 0), v)
    assert (smooth(np.array([1.0, 100, 1, 1]), 1.0, log=True) >= 0).all()


def test_smoothing_raises_correlation_of_noisy_signals():
    from loopsage.validation.compare import _compare, ValidationResult
    rng = np.random.default_rng(0)
    base = np.convolve(rng.random(300), np.ones(15) / 15, "same") * 10                      # smooth shared structure
    exp, sim = base + rng.normal(0, 1.5, 300) + 5, base + rng.normal(0, 1.5, 300) + 5
    res = ValidationResult()
    _compare(res, "Test", "toy", "pearson", exp, sim, "bead", sigma=3.0)
    raw, sm = res.rows
    assert sm["label"].endswith("smoothed") and sm["pearson"] > raw["pearson"] + 0.05
    assert res.plots[0]["smooth"] is not None and res.plots[0]["sigma"] == 3.0
    off = ValidationResult()
    _compare(off, "Test", "toy", "pearson", exp, sim, "bead", sigma=0.0)
    assert len(off.rows) == 1 and off.plots[0]["smooth"] is None                     # 0 turns smoothing off
    loop = ValidationResult()
    _compare(loop, "Test", "toy", "pearson_direct", exp, sim, "loop", sigma=3.0)
    assert len(loop.rows) == 1                                                       # per-loop values are never smoothed
