# tests/conftest.py
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create tiny synthetic test files once per session."""
    d = tmp_path_factory.mktemp("data")

    # Tiny bedpe (5 loops on chr1:0-10000)
    lines = []
    for i in range(5):
        s1, e1 = i * 1000, i * 1000 + 500
        s2, e2 = (i + 3) * 1000, (i + 3) * 1000 + 500
        lines.append(f"chr1\t{s1}\t{e1}\tchr1\t{s2}\t{e2}\t{10 + i}\t0.2\t0.8")
    (d / "tiny.bedpe").write_text("\n".join(lines))

    # Tiny narrowPeak (8 peaks)
    lines = []
    for i in range(8):
        s = i * 1000
        lines.append(f"chr1\t{s}\t{s+500}\tpeak_{i}\t{100+i*10}\t.\t5.0\t10.0\t8.0\t250")
    (d / "tiny.narrowPeak").write_text("\n".join(lines))

    # Tiny bed (6 intervals)
    lines = []
    for i in range(6):
        s = i * 1500
        lines.append(f"chr1\t{s}\t{s+400}\tsite_{i}\t{50+i}\t.")
    (d / "tiny.bed").write_text("\n".join(lines))

    return d

@pytest.fixture
def random_matrix():
    """A small symmetric 'simulated heatmap' for correlation tests."""
    rng = np.random.default_rng(42)
    N = 50
    m = rng.random((N, N))
    m = (m + m.T) / 2
    decay = np.array([[1.0 / (1 + abs(i - j)) for j in range(N)] for i in range(N)])
    return m * decay * 10
