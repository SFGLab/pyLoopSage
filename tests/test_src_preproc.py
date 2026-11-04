"""Input readers at bead resolution."""
import numpy as np
import pytest

from preproc.loader import load_inputs
from preproc.loader import build_geometry
from settings.loader import load_config


def _inputs(args):
    cfg = load_config(args)
    geom = build_geometry(cfg)
    return cfg, geom, load_inputs(cfg, geom)


def test_bedpe_shapes_and_stats(tiny_args):
    _, geom, inp = _inputs(tiny_args)
    lo = inp.loops
    assert lo.L.shape == lo.R.shape == (geom.n_beads,)
    assert lo.J.shape == lo.J_loss.shape == (geom.n_beads, geom.n_beads)
    assert lo.fmt == "bedpe" and lo.stats["n_loops"] > 0 and len(lo.loop_lengths) > 0
    assert np.all(lo.L >= 0) and np.all(lo.R >= 0)


def test_narrowpeak_has_no_loop_lengths(tmp_path, tiny_args):
    f = tmp_path / "p.narrowPeak"
    f.write_text("\n".join(f"chr1\t{77_000_000 + i * 900_000}\t{77_000_000 + i * 900_000 + 500}\tp{i}\t{100 + i}\t.\t5\t10\t8\t250"
                           for i in range(15)))
    args = list(tiny_args)
    args[1] = str(f)
    _, _, inp = _inputs(args)
    assert inp.loops.fmt != "bedpe" and len(inp.loops.loop_lengths) == 0 and inp.loops.n_sites > 0


def test_unsupported_extension(tmp_path, tiny_args):
    f = tmp_path / "x.txt"
    f.write_text("chr1\t1\t2\n")
    with pytest.raises(SystemExit):
        load_config([tiny_args[0], str(f)] + tiny_args[2:])


def test_input_cache_roundtrip(tiny_args, tmp_path):
    from preproc.loader import InputData
    _, _, inp = _inputs(tiny_args)
    inp.save(tmp_path / "in.npz")
    back = InputData.load(tmp_path / "in.npz")
    assert np.allclose(back.loops.L, inp.loops.L) and back.loops.stats == inp.loops.stats


def test_region_fallback_to_whole_chromosome(demo_bedpe):
    cfg = load_config(["-f", str(demo_bedpe), "--set", "chrom=chr21", "-n", "50"])
    geom = build_geometry(cfg)
    assert geom.start == 0 and geom.end > 40_000_000
