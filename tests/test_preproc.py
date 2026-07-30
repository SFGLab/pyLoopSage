# tests/test_preproc.py
import numpy as np

def test_bedpe_loading(test_data_dir):
    """binding_vectors_from_bedpe should return arrays of the right shape."""
    from loopsage.preproc import binding_vectors_from_bedpe

    L, R, J, J_loss, stats = binding_vectors_from_bedpe(
        bedpe_file=str(test_data_dir / "tiny.bedpe"),
        N_beads=20, region=[0, 10000], chrom="chr1",
        normalization=False, viz=False, diagonal_interactions=True,
        smooth=False, contrastive=False
    )
    assert L.shape == (20,)
    assert R.shape == (20,)
    assert J.shape == (20, 20)
    assert stats["n_loops"] > 0

def test_narrowpeak_has_no_loop_length(test_data_dir):
    """narrowPeak input should NOT produce loop_length stats (no anchor pairs)."""
    from loopsage.preproc import binding_vectors_from_narrowpeak

    L, R, J, J_loss, stats = binding_vectors_from_narrowpeak(
        narrowpeak_file=str(test_data_dir / "tiny.narrowPeak"),
        N_beads=20, region=[0, 10000], chrom="chr1",
        normalization=False, viz=False, diagonal_interactions=True,
        smooth=False, contrastive=False
    )
    assert "n_peaks" in stats
    assert "loop_length" not in stats

def test_format_detection():
    """Auto-detection should work for all three extensions."""
    from loopsage.preproc import detect_interaction_format
    # If detect_interaction_format lives in corr_exp_heat instead,
    # adjust the import accordingly.
    assert detect_interaction_format("foo.bedpe") == "bedpe"
    assert detect_interaction_format("foo.narrowPeak") == "narrowpeak"
    assert detect_interaction_format("foo.bed") == "bed"
