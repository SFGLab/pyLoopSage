"""Input readers at bead resolution."""
import numpy as np
import pytest

from loopsage.preproc.loader import load_inputs
from loopsage.preproc.loader import build_geometry
from loopsage.settings.loader import load_config


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
    from loopsage.preproc.loader import InputData
    _, _, inp = _inputs(tiny_args)
    inp.save(tmp_path / "in.npz")
    back = InputData.load(tmp_path / "in.npz")
    assert np.allclose(back.loops.L, inp.loops.L) and back.loops.stats == inp.loops.stats


def test_region_fallback_to_whole_chromosome(demo_bedpe):
    cfg = load_config(["-f", str(demo_bedpe), "--set", "chrom=chr21", "-n", "50"])
    geom = build_geometry(cfg)
    assert geom.start == 0 and geom.end > 40_000_000


def test_loss_map_is_smooth_only_in_2d_mode(tiny_args):
    raw = _inputs(tiny_args + ["-s", "binding_mode=preferential", "-s", "smoothing_input=False"])[2].loops.J_loss
    two_d = _inputs(tiny_args + ["-s", "data_loss_mode=1", "-s", "binding_mode=preferential", "-s", "smoothing_input=False"])[2].loops.J_loss
    assert (two_d > 0).sum() > 3 * (raw > 0).sum()        # sparse dots become a continuous field


def test_contrastive_adds_loop_coverage(tiny_args):
    pref = _inputs(tiny_args + ["--binding-mode", "preferential"])[2].loops
    cont = _inputs(tiny_args + ["--binding-mode", "contrastive"])[2].loops
    assert pref.cover is None and cont.cover is not None
    assert cont.cover.min() >= -1 and cont.cover.max() <= 1
    assert cont.cover.min() < 0 < cont.cover.max()          # loop-free beads negative, covered beads positive
    assert set(np.unique(cont.cover[cont.cover < 0])) == {-1.0}   # only loop-free beads are penalised
    assert np.allclose(pref.L, cont.L)                      # the anchor profiles are the same in both modes


def test_contrast_cover_penalises_only_loop_free_beads():
    from loopsage.preproc.readers import contrast_cover
    c = contrast_cover(np.array([0, 0, 1, 2, 5, 10, 10.0]))      # mean 4: beads with 1 or 2 loops are sparse, not empty
    assert list(c[:2]) == [-1, -1] and (c[2:] >= 0).all() and c[-1] == 1.0


# ---- BigWig binding tracks: smoothing, cleaning and the energy ----

def _bigwig(path, values, start=77_000_000, step=10_000):
    pybw = pytest.importorskip("pyBigWig")
    bw = pybw.open(str(path), "w")
    bw.addHeader([("chr1", 249_250_621)])
    n = len(values)
    bw.addEntries(["chr1"] * n, [start + i * step for i in range(n)], ends=[start + (i + 1) * step for i in range(n)], values=[float(v) for v in values])
    bw.close()
    return path


def _peaks(n=1800, centres=(400, 1300)):
    sig = np.zeros(n)
    for c in centres:
        sig[c - 15:c + 15] = 1.0
    return sig


def test_binding_track_is_smoothed_like_l_and_r(demo_bedpe, tmp_path):
    from loopsage.preproc.loader import load_inputs
    bw = _bigwig(tmp_path / "t.bw", _peaks() + 0.01)
    args = ["-f", str(demo_bedpe), "-r", "chr1:77000000-95000000", "-n", "120", "--set", f"bw_files={bw}"]
    on = load_inputs(load_config(args), build_geometry(load_config(args)))
    off_args = args + ["--set", "smoothing_input=False"]
    off = load_inputs(load_config(off_args), build_geometry(load_config(off_args)))
    assert on.BWs[0].max() < off.BWs[0].max()                                          # the peaks are spread over neighbouring beads
    assert np.isclose(on.BWs[0].sum(), off.BWs[0].sum())                               # the total signal is kept


def test_binding_track_cleaning():
    from loopsage.preproc.readers import binding_track
    v = np.array([0.0, 1.0, -0.2, 0.5, 0.0])
    assert binding_track(v).min() == 0 and binding_track(v, 1.0).sum() > 0
    with pytest.raises(ValueError, match="no signal"):
        binding_track(np.zeros(10), 1.0, "empty.bw")


def test_binding_track_energy_pulls_legs_to_the_signal(demo_bedpe, tmp_path):
    import loopsage as ls
    bw = _bigwig(tmp_path / "peaks.bw", _peaks() + 0.01)
    common = dict(region="chr1:77000000-95000000", n_beads=120, n_steps=12000, mc_step=200, burnin=2400, n_lef=6, quiet=True, seed=1)
    legs = {}
    for name, strength in (("none", None), ("strong", -300.0)):
        cfg = ls.config(**common, **({} if strength is None else {"bw_strengths": [strength]}))
        geom = ls.geometry(cfg)
        inputs = ls.read_inputs(cfg, geom, loops=str(demo_bedpe), tracks=None if strength is None else str(bw))
        mc = ls.run_mc(cfg, inputs, geom)
        assert np.isfinite(mc.Es).all()
        density = ls.results.leg_density(mc, geom)
        legs[name] = sum(density[p - 1:p + 2].sum() for p in (26, 86))                 # beads of the two peaks
    assert legs["strong"] > 1.3 * legs["none"]


def test_binding_energy_update_matches_the_full_energy():
    from loopsage.monte_carlo.energy import dE_bw, E_bw
    rng = np.random.default_rng(0)
    tracks, r = rng.random((2, 60)), np.array([-6.0, -3.0])
    ms, ns = np.array([5, 20]), np.array([30, 50])
    moved = ms.copy()
    moved[0] = 40
    assert np.isclose(dE_bw(r, tracks, 5, 30, 40, 30), E_bw(r, tracks, moved, ns) - E_bw(r, tracks, ms, ns))
