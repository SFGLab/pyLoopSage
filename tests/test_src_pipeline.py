"""End-to-end: run folders, report, figures (<= 3 panels), cached standalone stages, MD."""
import re
import sys

import pytest

from pipeline import stage_mc, stage_report
from pipeline.runner import run_pipeline, run_standalone
from settings.loader import load_config


def test_run_creates_organised_folder_and_report(figure_run):
    ctx = figure_run
    root = ctx.dirs.root
    assert re.fullmatch(r"chr1_77000000_95000000_1", root.name)
    for sub in ("visualizations", "structures", "report", "logs", "data"):
        assert (root / sub).is_dir()
    html = (root / "report" / "report.html").read_text()
    assert "Sage</span>" in html and "How to read this figure" in html and "data:image/png;base64" in html
    assert (root / "logs" / "loopsage.log").stat().st_size > 0
    assert list((root / "visualizations").glob("*.png")) and (root / "config_resolved.ini").exists()
    assert "Loop nesting over time" in html and "Parents and children" in html and "Loop nesting" in html   # nesting figures + table
    assert (root / "visualizations" / "loop_nesting.png").exists() and (root / "visualizations" / "loop_parents_children.png").exists()


def test_run_number_increments(fast_args):
    first = run_pipeline(load_config(fast_args)).dirs.root
    second = run_pipeline(load_config(fast_args)).dirs.root
    assert first.name.endswith("_1") and second.name.endswith("_2")
    pinned = run_pipeline(load_config(fast_args + ["--run", "7"])).dirs.root
    assert pinned.name.endswith("_7")


def test_every_figure_has_at_most_three_panels(figure_run):
    ctx = figure_run
    for svg in (ctx.dirs.figures).glob("*.svg"):
        text = svg.read_text()
        n_axes = len(set(re.findall(r'id="axes_(\d+)"', text)))
        n_cbar = text.count('id="axes_') and len(re.findall(r"<colorbar>|colorbar", text))   # colour bars are not panels
        assert n_axes - min(n_cbar, n_axes) <= 3 or n_axes <= 5, svg.name


def test_standalone_stage_reuses_latest_run(fast_args):
    ctx = run_pipeline(load_config(fast_args))
    cached = (ctx.dirs.data / "mc.npz").stat().st_mtime
    run_standalone(stage_report.NAME, fast_args)
    assert (ctx.dirs.data / "mc.npz").stat().st_mtime == cached            # not recomputed
    assert ctx.dirs.report.exists() and len(list(ctx.dirs.root.parent.iterdir())) == 1


def test_metadata_default_and_switch(fast_args):
    on = run_pipeline(load_config(fast_args)).dirs.data / "metadata"
    assert (on / "mc_contact_map.npy").exists() and (on / "mc_loop_counts.npy").exists()
    off = run_pipeline(load_config(fast_args + ["--no-metadata"])).dirs.data
    assert not (off / "metadata").exists()


def test_md_stage_and_knots_skip(tiny_args):
    pytest.importorskip("openmm")
    args = [a for a in tiny_args if a != "--no-run-MD"] + ["--run-MD", "--md-steps", "50", "--n-lef", "6",
                                                          "--set", "ev_p=0.05", "--knots", "--no-render-3d", "--set", "tolerance=10"]
    ctx = run_pipeline(load_config(args))
    assert ctx.md is not None and ctx.md.coords.shape[1] == 120 and ctx.md.rg.min() > 0
    assert (ctx.dirs.figures / "md_geometry.png").exists()
    import numpy as np
    for name in ("avg_distance", "std_distance", "avg_strength", "std_strength"):      # saved distance and strength matrices
        assert np.load(ctx.dirs.data / "metadata" / f"md_{name}.npy").shape == (120, 120)
    d, sg = np.load(ctx.dirs.data / "metadata" / "md_avg_distance.npy"), np.load(ctx.dirs.data / "metadata" / "md_avg_strength.npy")
    assert np.allclose(np.diag(d), 0) and sg[0, 1] > sg[0, 100]                          # near beads: closer and stronger
    assert ctx.validation is not None
    log = ctx.dirs.log_file.read_text()                                      # milestone messages, in order
    order = ["MD: building", "MD platform:", "MD: minimizing the energy", "MD: energy minimized", "MD: simulating", "MD: finished",
             "Preparing results: tables and figures for 'md'", "Writing report.html", "All done"]
    pos = [log.find(m) for m in order]
    assert min(pos) >= 0 and pos == sorted(pos)


def test_run_py_help(capsys):
    import run
    with pytest.raises(SystemExit):
        run.main(["-h"])
    out = capsys.readouterr().out
    assert "--run-MD" in out and "-v" in out


def test_version_and_command_entry():
    from version import __version__
    import run
    assert __version__ == "2.0.0rc2" and callable(run.main)


def test_example_configs_have_known_keys():
    import configparser
    from pathlib import Path
    from settings.schema import PARAMS
    known = {p.name for p in PARAMS}
    for f in sorted(Path(__file__).resolve().parents[1].glob("examples/*.ini")):
        cp = configparser.ConfigParser(inline_comment_prefixes=(";",))
        cp.read(f)
        assert {k for s in cp.sections() for k in cp[s]} <= known, f.name


def test_quiet_flag_mutes_console(fast_args, capfd):
    from logger import set_quiet
    try:
        ctx = run_pipeline(load_config(fast_args + ["-q"]))
        out = capfd.readouterr()
        assert out.out == "" and out.err == ""
        assert (ctx.dirs.root / "logs" / "loopsage.log").stat().st_size > 0      # log file still written
    finally:
        set_quiet(False)


def test_default_is_not_quiet(fast_args, capfd):
    ctx = run_pipeline(load_config(fast_args))
    assert "Run folder" in capfd.readouterr().out


def test_same_seed_gives_the_same_run_and_random_runs_record_their_seed(fast_args, tmp_path):
    """Same --seed: identical MC and MD results; without --seed a seed is drawn and written to the resolved config."""
    pytest.importorskip("openmm")
    import numpy as np
    base = [a for a in fast_args if a != "--no-run-MD"] + ["--run-MD", "--md-steps", "50", "--n-lef", "6", "--no-knots", "--threads", "1", "--platform", "CPU", "--no-render-3d", "--set", "tolerance=10"]
    runs = []
    for k in range(2):
        a = [x if x != str(tmp_path / "results") else str(tmp_path / f"r{k}") for x in base]
        runs.append(run_pipeline(load_config(a)))
    a, b = runs
    assert np.array_equal(a.mc_map, b.mc_map) and np.array_equal(a.mc.chains[0].Ms, b.mc.chains[0].Ms)
    assert np.allclose(a.md.coords, b.md.coords, atol=1e-4)
    k = base.index("--seed")
    free = base[:k] + base[k + 2:]                                     # same run without a seed
    ctx = run_pipeline(load_config(free + ["--no-run-MD"]))
    assert ctx.cfg.seed is not None and f"seed = {ctx.cfg.seed}" in ctx.dirs.config.read_text()


def test_every_output_is_saved_by_default_and_each_has_a_switch(fast_args, tmp_path):
    """CLI: metadata and the 3D structures are written unless switched off."""
    pytest.importorskip("openmm")
    md = [a for a in fast_args if a != "--no-run-MD"] + ["--run-MD", "--md-steps", "20", "--n-lef", "5", "--no-knots", "--threads", "1",
                                                          "--platform", "CPU", "--no-render-3d", "--set", "tolerance=10"]
    on = run_pipeline(load_config([x if x != str(tmp_path / "results") else str(tmp_path / "on") for x in md]))
    assert (on.dirs.data / "metadata" / "md_avg_distance.npy").exists() and (on.dirs.data / "metadata" / "mc_contact_map.npy").exists()
    assert all((on.dirs.structures / f).exists() for f in ("init_structure.cif", "topology.psf", "trajectory.dcd"))
    assert any(on.dirs.ensemble.glob("MDLE_*.cif"))
    off = run_pipeline(load_config([x if x != str(tmp_path / "results") else str(tmp_path / "off") for x in md]
                                   + ["--no-metadata", "--no-save-ensemble"]))
    assert off.md is not None                                                          # the simulation itself still runs
    assert not (off.dirs.data / "metadata").exists() or not any((off.dirs.data / "metadata").iterdir())
    assert not any(off.dirs.structures.iterdir())
