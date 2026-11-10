"""Python interface: `import loopsage as ls`, sub-package aliases, config from Python, the Simulation class."""
import os
import numpy as np
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def params(demo_bedpe, tmp_path):
    """Keyword parameters of a tiny run (no MD, no figures)."""
    return dict(interaction_file=str(demo_bedpe), region="chr1:77000000-95000000", n_beads=120, out_path=str(tmp_path / "results"),
                n_steps=2000, mc_step=100, burnin=400, run_md=False, seed=1, figures=False, quiet=True)


def _python(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_import_is_light():
    r = _python("import sys, loopsage; print(any(m in sys.modules for m in ('numba', 'openmm', 'matplotlib')), loopsage.__version__)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "False"


def test_submodules_are_the_same_objects():
    import loopsage as ls
    import loopsage.monte_carlo as mc
    from loopsage.preproc.loader import load_inputs
    assert ls.monte_carlo is mc and ls.preproc.loader.load_inputs is load_inputs
    with pytest.raises(ImportError):
        import loopsage.not_a_module  # noqa: F401


def test_top_level_names():
    import loopsage as ls
    from loopsage.settings.loader import Config
    assert ls.Config is Config and callable(ls.run_mc) and callable(ls.load_config) and callable(ls.read_inputs)


def test_config_from_python(params):
    import loopsage as ls
    cfg = ls.config(**params)
    assert (cfg.chrom, cfg.region_start, cfg.n_steps, cfg.run_md) == ("chr1", 77000000, 2000, False)
    assert cfg.source["n_steps"] == "cli" and cfg.source["tolerance"] == "default"
    assert cfg.command.startswith("loopsage -r chr1:77000000-95000000")


def test_config_from_ini_config_and_dict(params, tmp_path):
    import loopsage as ls
    from loopsage.settings.loader import write_ini
    base = ls.config(**params)
    ini = tmp_path / "c.ini"
    write_ini(ini, base.as_dict(), base.source)
    assert ls.config(ini).n_steps == 2000
    assert ls.config(ini, n_steps=3000, burnin=500).n_steps == 3000                  # parameters win over the file
    assert ls.config(base, n_steps=2500).seed == 1                                    # a Config can be reused
    assert ls.config({**params, "n_steps": 2200}).n_steps == 2200


def test_config_errors_raise_value_error(params):
    import loopsage as ls
    with pytest.raises(ValueError, match="unknown parameter"):
        ls.config(**params, not_a_parameter=1)
    with pytest.raises(ValueError, match="burn-in"):
        ls.config(**{**params, "burnin": 5000})
    with pytest.raises(ValueError, match="not found"):
        ls.config(interaction_file="missing.bedpe", region="chr1:1-100000")


def test_python_dash_m_loopsage():
    r = subprocess.run([sys.executable, "-m", "loopsage", "-h"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "usage: loopsage" in r.stdout


def test_python_dash_m_loopsage():
    r = subprocess.run([sys.executable, "-m", "loopsage", "-h"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "usage: loopsage" in r.stdout


REGION = "chr1:77000000-95000000"
LEFT = np.array([77_500_000, 80_000_000, 83_000_000, 88_000_000])
RIGHT = LEFT + np.array([3_000_000, 4_000_000, 2_500_000, 5_000_000])
WEIGHT = [10, 20, 5, 8]


@pytest.fixture
def small(tmp_path, monkeypatch):
    """A tiny configuration; the working folder is empty so stray files are visible."""
    import loopsage as ls
    monkeypatch.chdir(tmp_path)
    return ls.config(region=REGION, n_beads=120, n_steps=2000, mc_step=100, burnin=400, seed=1, n_lef=6, run_md=False, quiet=True,
                     sim_step=50, platform="CPU", threads=1, tolerance=10)


@pytest.fixture
def mc_run(small, demo_bedpe):
    import loopsage as ls
    geom = ls.geometry(small)
    inputs = ls.read_inputs(small, geom, loops=str(demo_bedpe))
    return small, geom, inputs, ls.run_mc(small, inputs, geom)


def test_functions_write_nothing_and_are_reproducible(mc_run, tmp_path):
    import loopsage as ls
    cfg, geom, inputs, mc = mc_run
    assert inputs.loops.L.shape == (120,) and mc.n_samples == 20 and mc.Ms.shape == (6, 20)
    assert np.array_equal(ls.run_mc(cfg, inputs, geom).Es, mc.Es)                       # same seed, same result
    assert list(tmp_path.iterdir()) == []


def test_results_functions(mc_run):
    import loopsage as ls
    r = ls.results
    cfg, geom, inputs, mc = mc_run
    left, right = r.trajectories(mc, geom, "mb")
    assert left.shape == (6, 20) and (right >= left).all() and 77 <= left.min() and right.max() <= 95
    assert r.loop_lengths(mc, geom, "bead").shape == (6, 20) and r.mean_loop_length(mc, geom).shape == (20,)
    assert r.mean_loop_length(mc, geom, production=True).shape == (20 - mc.burnin_idx,)
    e = r.energies(mc)
    assert list(e.columns) == ["step", "total", "folding", "binding", "crossing", "temperature", "unfolded_fraction"] and len(e) == 20
    assert len(r.loops_table(mc, geom)) == 6 and r.loop_coverage(mc, geom).shape == (120,) and r.leg_density(mc, geom).sum() > 0
    assert r.contact_map(mc, geom).shape == (120, 120) and r.steps(mc)[1] == 100


def test_plots_return_figures(mc_run, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import loopsage as ls
    cfg, geom, inputs, mc = mc_run
    assert isinstance(ls.plots.energy_traces(mc), matplotlib.figure.Figure)
    assert isinstance(ls.plots.binding_profiles(inputs, geom), matplotlib.figure.Figure)
    with pytest.raises(AttributeError):
        ls.plots.not_a_figure
    assert list(tmp_path.iterdir()) == []


def test_optimize_and_apply(small, demo_bedpe):
    import loopsage as ls
    cfg = ls.config(small, opt_evals=6, opt_steps=500)
    geom = ls.geometry(cfg)
    inputs = ls.read_inputs(cfg, geom, loops=str(demo_bedpe))
    res = ls.optimize(cfg, inputs, geom)
    new = ls.apply_optimum(cfg, res)
    assert new.n_lef == res.best["K"] and new.folding_coeff == res.best["f"] and cfg.folding_coeff != 0 and new is not cfg


def test_arrays_match_the_bedpe_file(small, tmp_path):
    import loopsage as ls
    geom = ls.geometry(small)
    f = tmp_path / "l.bedpe"
    f.write_text("".join(f"chr1\t{a}\t{a + 1000}\tchr1\t{b}\t{b + 1000}\t{w}\n" for a, b, w in zip(LEFT, RIGHT, WEIGHT)))
    a = ls.read_inputs(small, geom, loops=str(f)).loops
    b = ls.read_inputs(small, geom, loops=dict(left=LEFT + 500, right=RIGHT + 500, weight=WEIGHT)).loops
    assert np.allclose(a.L, b.L) and np.allclose(a.R, b.R) and np.array_equal(a.J, b.J)
    assert ls.read_inputs(small, geom, loops=(LEFT, RIGHT, WEIGHT)).loops.stats["n_loops"] == 4


def test_extra_inputs_as_arrays_and_checks(small):
    import loopsage as ls
    geom = ls.geometry(small)
    inp = ls.read_inputs(small, geom, loops=(LEFT, RIGHT), tracks=[np.ones(120)], lef_track=np.arange(120), compartments=np.linspace(-1, 1, 120))
    assert inp.BWs.shape == (1, 120) and inp.lef_track.max() == 119 and inp.comp.min() == -1 and inp.comp_source == "array"
    with pytest.raises(ValueError, match="between -1 and 1"):
        ls.read_inputs(small, geom, loops=(LEFT, RIGHT), compartments=np.linspace(-10, 10, 120))
    with pytest.raises(ValueError, match="one per bead"):
        ls.read_inputs(small, geom, loops=(LEFT, RIGHT), tracks=np.zeros((1, 50)))
    with pytest.raises(ValueError, match="give the loops"):
        ls.read_inputs(small, geom)


def test_molecular_dynamics(mc_run, tmp_path):
    pytest.importorskip("openmm")
    import loopsage as ls
    cfg, geom, inputs, mc = mc_run
    md = ls.run_md(cfg, mc, geom)
    n = md.coords.shape[0]
    assert ls.results.coordinates(md).shape == (n, 120, 3) and ls.results.coordinates(md, 0).shape == (120, 3)
    assert ls.results.radius_of_gyration(md).shape == (n,) and ls.results.distance(md, geom, 0, 100).shape == (n,)
    assert ls.results.contact_map_md(md).shape == (120, 120)
    assert list(tmp_path.iterdir()) == []                                                 # no folder given: nothing written


def test_sandbox(small):
    import loopsage as ls
    box = ls.Sandbox("chr1:0-3000000", n_beads=100)
    box.add_ctcfs([300_000, 1_500_000], "forward", [10, 6]).add_ctcf(900_000, "reverse", 8).add_ctcf(2_400_000, 0.25)
    assert list(box.ctcf_sites["position"]) == [300_000, 1_500_000, 900_000, 2_400_000]
    with pytest.raises(ValueError, match="between -1 and 1"):
        box.set_compartments(np.linspace(-10, 10, 100))
    with pytest.raises(ValueError, match="expected 100"):
        box.set_compartments(np.zeros(50))
    with pytest.raises(ValueError, match="outside"):
        box.add_ctcf(5_000_000)
    box.set_compartments(np.sin(np.linspace(0, 12, 100)))
    box.add_compartment(0, 500_000, "B")
    assert box.compartments.shape == (100,) and box.compartments[0] == -1
    assert box.random_compartments(0.5, seed=1).compartments.shape == (100,)
    cfg = box.config(n_steps=3000, mc_step=100, burnin=500, seed=2, quiet=True, n_lef=4)
    geom = ls.geometry(cfg)
    inputs = ls.read_inputs(cfg, geom, **box.arrays())
    assert inputs.loops.fmt == "bed" and inputs.comp_source == "array" and ls.run_mc(cfg, inputs, geom).n_samples == 30
    loops = ls.Sandbox("chr1:0-3000000", n_beads=100).add_loop(300_000, 900_000, 5).add_loop(1_500_000, 2_400_000, 3)
    assert ls.read_inputs(cfg, geom, **loops.arrays()).loops.stats["n_loops"] == 2
    with pytest.raises(ValueError):
        ls.Sandbox().arrays()


def test_parameters_helper():
    import loopsage as ls
    names = ls.parameters("md steps")
    assert "sim_step" in names and ls.parameters()["n_steps"][0] == 20000


def test_n_sweep_default_is_two_per_lef():
    """n_sweep empty = 2 x LEFs (same chain as setting it explicitly); a smaller value changes the chain."""
    import numpy as np
    import loopsage as ls
    run = {}
    for sw in (None, 16, 4):
        kw = dict(region="chr14:22500000-24000000", n_beads=120, n_steps=600, mc_step=100, burnin=200, seed=1, quiet=True, n_lef=8, threads=1)
        if sw:
            kw["n_sweep"] = sw
        cfg = ls.config(**kw)
        geom = ls.geometry(cfg)
        run[sw] = ls.run_mc(cfg, ls.read_inputs(cfg, geom, loops="tests/data/tiny.bedpe"), geom)
    assert np.array_equal(run[None].Ms, run[16].Ms)
    assert not np.array_equal(run[None].Ms, run[4].Ms)
