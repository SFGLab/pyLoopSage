"""Signal normalisation and the correlation of two signals (zero/zero pairs are left out)."""
import numpy as np

from common.utils import normalize_signal
from validation.compare import correlate


def test_zero_exclusion_changes_result():
    """Positions where both signals are zero agree trivially: counting them inflates r, so dropping them must change it."""
    exp = np.r_[np.zeros(30), [1, 2, 3, 4, 5]]
    sim = np.r_[np.zeros(30), [2, 1, 4, 3, 6]]
    kept, dropped = correlate(exp, sim, drop_zero=False), correlate(exp, sim, drop_zero=True)
    r = lambda d: next(v for k, v in d.items() if "pearson" in k.lower() and np.isscalar(v))
    assert np.isfinite(r(kept)) and np.isfinite(r(dropped))
    assert r(kept) > r(dropped)                                    # the shared zeros inflate the correlation


def test_normalize_signal_zero_std():
    """A constant vector must not crash and gives zeros (mean-centred, no variance to divide by)."""
    for log in (True, False):
        out = normalize_signal(np.ones(10), log_transform=log)
        assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)
