# tests/test_energy.py
import numpy as np
from loopsage.stochastic_simulation import (
    Kappa, E_bind, E_fold, E_cross, E_data,
    get_dE_bind, get_dE_fold
)

def test_kappa_no_crossing():
    """Two non-overlapping loops should have zero crossing."""
    assert Kappa(0, 5, 10, 15) == 0.0

def test_kappa_crossing():
    """mi < mj < ni < nj is a crossing."""
    assert Kappa(0, 10, 5, 15) == 1.0

def test_kappa_shared_endpoint():
    """Shared endpoint counts as a crossing."""
    assert Kappa(0, 10, 10, 20) == 1.0

def test_e_fold_increases_with_loop_size():
    """Longer loops should have higher (less negative) folding energy."""
    ms = np.array([0], dtype=np.int64)
    ns_short = np.array([5], dtype=np.int64)
    ns_long = np.array([50], dtype=np.int64)
    # fold_norm is negative, so larger log(n-m) -> more negative E
    assert E_fold(ms, ns_long, -1.0) < E_fold(ms, ns_short, -1.0)

def test_e_bind_uses_L_and_R():
    """E_bind should sum L[m] + R[n] for each LEF."""
    L = np.zeros(10)
    R = np.zeros(10)
    L[2] = 5.0
    R[7] = 3.0
    ms = np.array([2], dtype=np.int64)
    ns = np.array([7], dtype=np.int64)
    assert E_bind(L, R, ms, ns) == 8.0

def test_dE_bind_consistency():
    """dE should equal E(new) - E(old) for a single-LEF move."""
    L = np.arange(10, dtype=np.float64)
    R = np.arange(10, dtype=np.float64) * 0.5
    ms = np.array([2], dtype=np.int64)
    ns = np.array([7], dtype=np.int64)

    E_old = E_bind(L, R, ms, ns)
    dE = get_dE_bind(L, R, 1.0, ms, ns, 4, 8, 0)

    ms_new = np.array([4], dtype=np.int64)
    ns_new = np.array([8], dtype=np.int64)
    E_new = E_bind(L, R, ms_new, ns_new)

    assert abs(dE - (E_new - E_old)) < 1e-8

def test_e_cross_zero_for_nested_loops():
    """Properly nested loops (one inside the other) shouldn't cross."""
    ms = np.array([0, 5], dtype=np.int64)
    ns = np.array([20, 15], dtype=np.int64)
    assert E_cross(ms, ns, 1.0, 2, cross_loop=True) == 0.0
