"""Binding energy: the incremental change of the energy equals the difference of the totals.

The basic terms (kappa, folding, binding, crossing) are covered in test_src_energy.py.
"""
import numpy as np

from loopsage.monte_carlo.energy import E_bind, dE_data


def test_dE_data_matches_the_difference_of_e_bind():
    L, R = np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64) * 0.5
    old, new = (np.array([2]), np.array([7])), (np.array([4]), np.array([8]))
    d = dE_data(0, L, R, np.zeros((10, 10)), 1.0, 2, 7, 4, 8)
    assert abs(d - (E_bind(L, R, *new) - E_bind(L, R, *old))) < 1e-8
