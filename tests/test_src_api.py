"""Python interface: `import loopsage as ls`, sub-package aliases, config from Python, the Simulation class."""
import os
import numpy as np
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def params(demo_bedpe, tmp_path):
    """Keyword parameters of a tiny run (no MD, no figures)."""
    return dict(interaction_file=str(demo_bedpe), region="chr1:77000000-95000000", n_beads=120, out_path=str(tmp_path / "results"),
                n_steps=2000, mc_step=100, burnin=400, run_md=False, seed=1, figures=False, quiet=True)


def _python(code):
    env = dict(os.environ, PYTHONPATH=str(SRC))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)


def test_import_is_light():
    r = _python("import sys, loopsage; print(any(m in sys.modules for m in ('numba', 'openmm', 'matplotlib')), loopsage.__version__)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "False"


def test_submodules_are_the_same_objects():
    import loopsage.monte_carlo as mc
    import monte_carlo
    from loopsage.preproc.loader import load_inputs
    from preproc.loader import load_inputs as direct
    assert mc is monte_carlo and load_inputs is direct
    with pytest.raises(ImportError):
        import loopsage.not_a_module  # noqa: F401


def test_top_level_names():
    import loopsage as ls
    from settings.loader import Config
    assert ls.Config is Config and callable(ls.Simulation) and callable(ls.load_config) and callable(ls.run)


def test_config_from_python(params):
    import loopsage as ls
    cfg = ls.config(**params)
    assert (cfg.chrom, cfg.region_start, cfg.n_steps, cfg.run_md) == ("chr1", 77000000, 2000, False)
    assert cfg.source["n_steps"] == "cli" and cfg.source["tolerance"] == "default"
    assert cfg.command.startswith("loopsage -r chr1:77000000-95000000")


def test_config_from_ini_config_and_dict(params, tmp_path):
    import loopsage as ls
    from settings.loader import write_ini
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


def test_simulation_steps(params):
    import loopsage as ls
    sim = ls.Simulation(**params)
    inputs = sim.preprocess()
    assert inputs is sim.inputs and inputs.loops.L.shape == (120,)
    mc = sim.monte_carlo()                                                            # preprocessing is not repeated
    assert mc is sim.mc and mc.n_samples == 20 and sim.run_dir.is_dir()
    assert sim.analyze() is sim.validation
    assert Path(sim.report()).name == "report.html"
    with pytest.raises(AttributeError):
        sim.not_there


def test_simulation_step_runs_its_prerequisites(params):
    import loopsage as ls
    sim = ls.Simulation(**params)
    assert sim.monte_carlo().n_samples == 20 and sim.inputs is not None


def test_run_function_and_run_once(params):
    import loopsage as ls
    sim = ls.run(**params)
    assert sim.mc.n_samples == 20 and Path(sim.report_path).is_file()
    with pytest.raises(RuntimeError):
        sim.run()


def test_python_dash_m_loopsage():
    r = subprocess.run([sys.executable, "-m", "loopsage", "-h"], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(SRC)), timeout=120)
    assert r.returncode == 0 and "usage: loopsage" in r.stdout


# ---- Inputs and results as Python variables: no files ----------------------

REGION = "chr1:77000000-95000000"
LEFT = np.array([77_500_000, 80_000_000, 83_000_000, 88_000_000])
RIGHT = LEFT + np.array([3_000_000, 4_000_000, 2_500_000, 5_000_000])
WEIGHT = [10, 20, 5, 8]


@pytest.fixture
def mem(tmp_path):
    """Keyword parameters of a tiny in-memory run (no input file)."""
    return dict(region=REGION, n_beads=120, n_steps=2000, mc_step=100, burnin=400, seed=1, n_lef=6, sim_step=50,
                platform="CPU", threads=1, tolerance=10, quiet=True, out_path=str(tmp_path / "results"))


def test_loops_from_arrays_match_the_bedpe_reader(tmp_path):
    import loopsage as ls
    from preproc.readers import vectors_from_bedpe
    geom = ls.geometry("chr1", 77_000_000, 95_000_000, 120)
    f = tmp_path / "l.bedpe"
    f.write_text("".join(f"chr1\t{a}\t{a + 1000}\tchr1\t{b}\t{b + 1000}\t{w}\n" for a, b, w in zip(LEFT - 500, RIGHT - 500, WEIGHT)))
    a = ls.loops_from_arrays(geom, LEFT, RIGHT, WEIGHT)
    b = vectors_from_bedpe(f, geom)
    assert np.allclose(a.L, b.L) and np.allclose(a.R, b.R) and np.array_equal(a.J, b.J) and np.allclose(a.cover, b.cover)
    beads = ls.loops_from_arrays(geom, geom.bead_of(LEFT), geom.bead_of(RIGHT), WEIGHT, units="bead")
    assert np.allclose(beads.L, a.L)


def test_arrays_are_checked():
    import loopsage as ls
    geom = ls.geometry("chr1", 77_000_000, 95_000_000, 120)
    with pytest.raises(ValueError, match="same length"):
        ls.loops_from_arrays(geom, LEFT, RIGHT[:2])
    with pytest.raises(ValueError, match="weight has 2 values"):
        ls.loops_from_arrays(geom, LEFT, RIGHT, [1, 2])
    with pytest.raises(ValueError, match="probabilities"):
        ls.loops_from_arrays(geom, LEFT, RIGHT, left_reverse=[2, 0, 0, 0], right_reverse=[0, 0, 0, 0])
    with pytest.raises(ValueError, match="one per bead"):
        ls.inputs_from_arrays(geom, loops=(LEFT, RIGHT), tracks=np.zeros((1, 50)))
    with pytest.raises(ValueError, match="give loops or sites"):
        ls.inputs_from_arrays(geom)


def test_inputs_from_arrays_tracks_and_peaks():
    import loopsage as ls
    geom = ls.geometry("chr1", 77_000_000, 95_000_000, 120)
    inp = ls.inputs_from_arrays(geom, loops=dict(left=LEFT, right=RIGHT), tracks=[np.ones(120), np.zeros(120)],
                                lef_track=np.arange(120), compartments=np.linspace(-5, 5, 120))
    assert inp.BWs.shape == (2, 120) and inp.lef_track.max() == 119 and inp.comp.min() == -1 and inp.comp.max() == 1
    peaks = ls.inputs_from_arrays(geom, sites=dict(position=LEFT, weight=WEIGHT, reverse=[0, 1, 0.5, -1]))
    assert peaks.loops.fmt == "sites" and peaks.loops.L.shape == (120,)


def test_simulate_returns_variables_and_writes_nothing(mem, tmp_path, monkeypatch):
    import loopsage as ls
    monkeypatch.chdir(tmp_path)
    mc = ls.simulate(loops=dict(left=LEFT, right=RIGHT, weight=WEIGHT), **mem)
    assert mc.Ms.shape == (6, 20) and mc.Es.shape == (20,) and mc.meta["n_lef"] == 6
    assert ls.contact_map(mc).shape == (120, 120)
    assert list(tmp_path.iterdir()) == []                                              # no run folder, no files


def test_simulate_same_seed_same_result(mem):
    import loopsage as ls
    kw = dict(loops=dict(left=LEFT, right=RIGHT), **mem)
    a, b = ls.simulate(**kw), ls.simulate(**kw)
    assert np.array_equal(a.Ms, b.Ms) and np.array_equal(a.Es, b.Es)


def test_simulate_md_in_memory(mem, tmp_path, monkeypatch):
    pytest.importorskip("openmm")
    import loopsage as ls
    monkeypatch.chdir(tmp_path)
    mc = ls.simulate(loops=(LEFT, RIGHT), **mem)
    md = ls.simulate_md(mc, **mem)
    assert md.coords.shape == (20, 120, 3) and md.rg.min() > 0
    assert list(tmp_path.iterdir()) == []


def test_simulation_with_arrays(mem):
    import loopsage as ls
    sim = ls.Simulation(loops=dict(left=LEFT, right=RIGHT, weight=WEIGHT), **mem)
    assert sim.monte_carlo().n_samples == 20 and sim.inputs.loops.stats["n_loops"] == 4
    for step in (sim.analyze, sim.report, sim.run):
        with pytest.raises(ValueError, match="needs the loops as a file"):
            step()
    with pytest.raises(ValueError, match="give an input"):
        ls.Simulation(**mem)


def test_parameters_helper():
    import loopsage as ls
    names = ls.parameters("md steps")
    assert "sim_step" in names and ls.parameters()["n_steps"][0] == 20000
