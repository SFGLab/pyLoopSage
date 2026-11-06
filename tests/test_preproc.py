"""Input readers on tiny synthetic files (5 loops / 8 peaks on chr1:0-10000, 20 beads)."""
import pytest

from common.genome import Geometry
from preproc.readers import vectors_from_bedpe, vectors_from_narrowpeak
from validation.compare import detect_format


@pytest.fixture
def geom():
    return Geometry("chr1", 0, 10000, 20)


def test_bedpe_loading(test_data_dir, geom):
    lo = vectors_from_bedpe(str(test_data_dir / "tiny.bedpe"), geom, smooth=False, contrastive=False)
    assert lo.L.shape == (20,) and lo.R.shape == (20,) and lo.J.shape == (20, 20)
    assert lo.stats["n_loops"] > 0


def test_narrowpeak_has_no_loop_length(test_data_dir, geom):
    """narrowPeak input has no anchor pairs, so no loop-length statistics."""
    lo = vectors_from_narrowpeak(str(test_data_dir / "tiny.narrowPeak"), geom, smooth=False, contrastive=False)
    assert "n_peaks" in lo.stats and "loop_length" not in lo.stats


def test_format_detection():
    assert detect_format("foo.bedpe") == "bedpe"
    assert detect_format("foo.narrowPeak") == "narrowpeak"
    assert detect_format("foo.bed") == "bed"
    with pytest.raises(ValueError):
        detect_format("foo.txt")
