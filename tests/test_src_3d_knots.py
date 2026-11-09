"""Tests: 3D interpolation, structure metrics, viewer, map features and linking numbers."""
import sys
from pathlib import Path

import numpy as np
import pytest


from loopsage.analysis.maps import insulation, map_features, observed_over_expected   # noqa: E402
from loopsage.analysis.structure import compute_metrics                               # noqa: E402
from loopsage.knots.linking import frame_links, gauss_linking                                 # noqa: E402
from loopsage.molecular_dynamics.structures import interpolate_trajectory                    # noqa: E402


def ring(plane, center, r=1.0, n=40):
    """Circle in the xy or xz plane."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = r * np.cos(t), r * np.sin(t)
    pts = np.c_[c, s, 0 * t] if plane == "xy" else np.c_[c, 0 * t, s]
    return pts + np.asarray(center, float)


def test_interpolation_shape_and_smoothness():
    rng = np.random.default_rng(0)
    frames = np.cumsum(rng.normal(size=(6, 20, 3)), axis=1)
    out = interpolate_trajectory(frames, steps=4)
    assert out.shape == ((6 - 1) * 4 + 1, 20, 3)
    assert np.abs(np.diff(out, axis=0)).mean() < np.abs(np.diff(frames, axis=0)).mean()


def test_structure_metrics_rod_vs_blob():
    rod = np.c_[np.linspace(0, 10, 60), np.zeros(60), np.zeros(60)]
    blob = np.random.default_rng(1).normal(size=(200, 3))
    m_rod, m_blob = (compute_metrics(np.stack([x, x]), 1.0).shape for x in (rod, blob))
    assert m_rod["asphericity"].mean() > m_blob["asphericity"].mean()


def test_gauss_linking_hopf_and_unlinked():
    a, b = ring("xy", [0, 0, 0]), ring("xz", [1, 0, 0])
    assert abs(gauss_linking(a, b)) == pytest.approx(1.0, abs=1e-6)
    assert gauss_linking(a, ring("xy", [0, 0, 5])) == pytest.approx(0.0, abs=1e-6)


def test_frame_links_runs_on_chain():
    coords = np.cumsum(np.random.default_rng(2).normal(size=(60, 3)), axis=0)
    rows, n_pairs, n_loops, near = frame_links(coords, np.array([2, 20, 40]), np.array([15, 35, 55]))
    assert n_loops == 3 and n_pairs == 3 and near.shape == (3, 2) and rows.shape[1] == 5


def test_map_features():
    n = 40
    s = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    mat = 1.0 / (1 + s)
    mat[:20, :20] *= 3
    mat[20:, 20:] *= 3                                       # two domains
    f = map_features(mat)
    assert f.pc1.shape == (n,) and f.slope < 0
    assert np.all(np.abs(f.boundaries - 20) <= 3) or len(f.boundaries) == 0
    assert observed_over_expected(mat).shape == mat.shape
    score, _ = insulation(mat)
    assert np.nanargmin(score[10:30]) + 10 in range(17, 24)


def test_viewer_file(tmp_path):
    from loopsage.viz3d.render import build_viewer
    p = build_viewer(np.random.default_rng(0).normal(size=(3, 10, 3)), tmp_path / "v.html")
    t = p.read_text()
    assert "Tube (linear)" in t and "Beads on a string" in t


def test_loop_pairs_and_zoom_window():
    from loopsage.analysis.ensemble import loop_pairs_of_frame
    from types import SimpleNamespace
    from loopsage.viz.mc_figs import _zoom_window
    p = loop_pairs_of_frame(np.array([5, 3, 7, 9]), np.array([2, 8, 7, 12]))
    assert p.tolist() == [[2, 5], [3, 8], [9, 12]]            # dedup, ordered, zero-length dropped
    res = SimpleNamespace(n_samples=1000, burnin_idx=200)
    a, b = _zoom_window(res, 100)
    assert b - a == 100 and a >= 200


def test_bend_angle_straight_chain():
    from loopsage.analysis.structure import _bend
    c = np.c_[np.arange(6.0), np.zeros(6), np.zeros(6)]
    assert np.allclose(_bend(c), 180.0)


def test_replicas_threads_reproducible_and_pooled(tmp_path):
    from loopsage.analysis.ensemble import r_hat
    from loopsage.monte_carlo.simulation import MCResult
    from loopsage.monte_carlo.simulation import run_mc
    from loopsage.preproc.loader import load_inputs
    from loopsage.preproc.loader import build_geometry
    from loopsage.settings.loader import load_config
    root = Path(__file__).resolve().parents[1]
    base = ["-c", str(root / "demo/demo_quick.ini"), "-f", str(root / "demo/data/demo_loops.bedpe"), "--seed", "3", "--mc-steps", "2000", "--n-replicas", "3"]
    out = {}
    for t in (1, 3):
        cfg = load_config(base + ["--threads", str(t)])
        g = build_geometry(cfg)
        out[t] = run_mc(cfg, load_inputs(cfg, g), g)
    a, b = out[1], out[3]
    assert len(a.chains) == 3 and all(np.array_equal(x.Ms, y.Ms) for x, y in zip(a.chains, b.chains))   # thread count never changes results
    assert not np.array_equal(a.chains[0].Ms, a.chains[1].Ms)                                         # replicas differ
    pool = a.pooled()
    assert pool.meta["burnin"] == 0 and pool.n_samples == sum(c.n_samples - c.burnin_idx for c in a.chains)
    a.save(tmp_path / "mc.npz")
    assert len(MCResult.load(tmp_path / "mc.npz").replicas) == 2
    assert r_hat([np.random.default_rng(i).normal(size=200) for i in range(3)]) < 1.1
