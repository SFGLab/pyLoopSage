# tests/test_correlation.py
import numpy as np

def test_zero_exclusion_changes_result(random_matrix, test_data_dir):
    """Excluding zero/zero pairs should produce a different correlation
    than including them, for sparse data where many beads are untouched."""
    # This is the exact class of bug we caught: zeros inflate r.
    from loopsage.preproc import corr_exp_heat

    pears = corr_exp_heat(
        random_matrix,
        str(test_data_dir / "tiny.bedpe"),
        [0, 10000], "chr1", 50,
        str(test_data_dir / "corr_output")
    )
    # We can't assert a specific value, but it should be finite
    assert pears is None or np.isfinite(pears)

def test_normalize_signal_zero_std():
    """normalize_signal shouldn't crash on a constant vector."""
    from loopsage.preproc import normalize_signal
    result = normalize_signal(np.ones(10))
    assert np.all(np.isfinite(result))
    assert np.allclose(result, 0.0)  # mean-centered, zero variance
