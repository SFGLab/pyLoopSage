"""Small helpers: units, stats, matrices, geometry."""
import numpy as np
import pytest

from loopsage.common.genome import Geometry
from loopsage.common.genome import fmt_bp, fmt_duration, fmt_num
from loopsage.analysis.maps import insulation_score, minmax, ps_curve
from loopsage.common.utils import moving_average, normalize_signal, pearson_ci, strength_label
from loopsage.validation.compare import correlate


def test_formatting():
    assert fmt_bp(500) == "500 bp" and fmt_bp(45_000) == "45.0 kb" and fmt_bp(18e6) == "18.00 Mb"
    assert fmt_duration(5) == "5.0 s" and "min" in fmt_duration(125)
    assert fmt_num(1000) == "1,000"


def test_geometry():
    g = Geometry("chr1", 1_000_000, 2_000_000, 100)
    assert g.bp_per_bead == 10_000 and g.bead_of(1_500_000) == 50
    assert len(g.axis_mb()) == 100 and g.label.startswith("chr1:")


def test_normalize_signal_constant_vector():
    out = normalize_signal(np.ones(10))
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)


def test_moving_average_keeps_length_and_mean_of_constant():
    y = np.full(50, 3.0)
    assert np.allclose(moving_average(y, 7), 3.0) and len(moving_average(y, 7)) == 50
    assert np.array_equal(moving_average(y, 1), y)


def test_pearson_ci_and_label():
    lo, hi = pearson_ci(0.5, 100)
    assert lo < 0.5 < hi
    assert strength_label(0.05) == "negligible" and strength_label(0.8) == "very strong"


def test_correlate_perfect_and_degenerate():
    x = np.arange(20, dtype=float)
    c = correlate(x, 2 * x + 1, drop_zero=False)
    assert c["pearson"] == pytest.approx(1.0) and c["strength"] == "very strong"
    assert correlate(np.ones(10), x[:10]) is None


def test_zero_pairs_are_dropped():
    exp, sim = np.array([0, 0, 0, 1, 2, 3.0]), np.array([0, 0, 0, 3, 2, 1.0])
    assert correlate(exp, sim, drop_zero=True)["n"] == 3


def test_matrix_helpers():
    m = np.random.default_rng(0).random((30, 30))
    m = (m + m.T) / 2
    assert minmax(m).min() == 0 and minmax(m).max() == 1
    assert len(ps_curve(m)) == 30 and len(insulation_score(m)) == 30
