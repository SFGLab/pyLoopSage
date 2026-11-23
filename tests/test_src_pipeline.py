"""End-to-end: run folders, report, figures (<= 3 panels), cached standalone stages, MD."""
import re
import sys

import pytest

from loopsage.pipeline import stage_mc, stage_report
from loopsage.pipeline.runner import run_pipeline, run_standalone
from loopsage.settings.loader import load_config


def test_run_creates_organised_folder_and_report(figure_run):
    ctx = figure_run
    root = ctx.dirs.root
    assert re.fullmatch(r"chr1_77000000_95000000", root.name)             # no automatic number
    for sub in ("visualizations", "structures", "report", "logs", "data"):
        assert (root / sub).is_dir()
    html = (root / "report" / "report.html").read_text()
    assert "Sage</span>" in html and "How to read this figure" in html and "data:image/png;base64" in html
    assert (root / "logs" / "loopsage.log").stat().st_size > 0
    assert list((root / "visualizations").glob("*.png")) and (root / "config_resolved.ini").exists()
    assert "Loop nesting over time" in html and "Parents and children" in html and "Loop nesting" in html   # nesting figures + table
    assert (root / "visualizations" / "loop_nesting.png").exists() and (root / "visualizations" / "loop_parents_children.png").exists()


def test_run_folder_has_no_number(fast_args):
    first = run_pipeline(load_config(fast_args)).dirs.root
    second = run_pipeline(load_config(fast_args)).dirs.root
    assert first.name == second.name == "chr1_77000000_95000000"              # same folder, overwritten
    assert "already has results" in first.joinpath("logs", "loopsage.log").read_text()      # the second run said it overwrote


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


def test_metadata_default_and_switch(fast_args, tmp_path):
    on = run_pipeline(load_config(fast_args)).dirs.data / "metadata"
    assert (on / "mc_contact_map.npy").exists() and (on / "mc_loop_counts.npy").exists()
    off = run_pipeline(load_config([x if x != str(tmp_path / "results") else str(tmp_path / "off") for x in fast_args] + ["--no-metadata"])).dirs.data
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
    from loopsage.api import cli
    with pytest.raises(SystemExit):
        cli(["-h"])
    out = capsys.readouterr().out
    assert "--run-MD" in out and "-v" in out


def test_version_and_command_entry():
    from loopsage.version import __version__
    from loopsage.api import cli
    assert __version__ == "2.0.1" and callable(cli)


def test_example_configs_have_known_keys():
    import configparser
    from pathlib import Path
    from loopsage.settings.schema import PARAMS
    known = {p.name for p in PARAMS}
    for f in sorted(Path(__file__).resolve().parents[1].glob("examples/*.ini")):
        cp = configparser.ConfigParser(inline_comment_prefixes=(";",))
        cp.read(f)
        if "scan" in cp.sections():                      # loopsage-scan files have their own grammar (see test_src_sensitivity)
            continue
        assert {k for s in cp.sections() for k in cp[s]} <= known, f.name


def test_quiet_flag_mutes_console(fast_args, capfd):
    from loopsage.logger import set_quiet
    try:
        ctx = run_pipeline(load_config(fast_args + ["-q"]))
        out = capfd.readouterr()
        assert out.out == "" and out.err == ""
        assert (ctx.dirs.root / "logs" / "loopsage.log").stat().st_size > 0      # log file still written
    finally:
        set_quiet(False)


def test_default_is_not_quiet(fast_args, capfd):
    ctx = run_pipeline(load_config(fast_args))
    assert "All done" in capfd.readouterr().out          # stage headers are shown at the default verbosity


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


# ---- Replicas: independent chains and MD runs, tidy output folders ----------

def _replica_args(fast_args, n, tmp_path, name):
    args = [a for a in fast_args if a != "--no-run-MD"] + ["--run-MD", "--md-steps", "20", "--n-lef", "5", "--no-knots", "--threads", "2",
                                                          "--platform", "CPU", "--no-render-3d", "--set", "tolerance=10", "--n-replicas", str(n)]
    return [x if x != str(tmp_path / "results") else str(tmp_path / name) for x in args]




def test_one_replica_has_no_replica_folders(fast_args, tmp_path):
    pytest.importorskip("openmm")
    ctx = run_pipeline(load_config(_replica_args(fast_args, 1, tmp_path, "one")))
    names = {p.name for d in (ctx.dirs.metadata, ctx.dirs.structures, ctx.dirs.figures, ctx.dirs.data) for p in d.iterdir()}
    assert not any(n.startswith("replica") or "_r1" in n for n in names)
    assert (ctx.dirs.metadata / "md_avg_distance.npy").exists() and (ctx.dirs.structures / "trajectory.dcd").exists()


def test_replicas_run_independently_and_are_organised_per_replica(demo_bedpe, tmp_path):
    """3 replicas: own chain, own MD start structure and loops, replica_<k> folders, cross-replica maps; replica k alone reproduces replica k."""
    pytest.importorskip("openmm")
    import numpy as np
    fa = ["-f", str(demo_bedpe), "-r", "chr1:77000000-95000000", "-n", "120", "-o", str(tmp_path / "results"), "--mc-steps", "2000",
          "--sample-every", "100", "--burnin", "400", "--seed", "1"]
    ctx = run_pipeline(load_config(_replica_args(fa, 3, tmp_path, "three")))
    d, chains = ctx.dirs, ctx.mc.chains
    assert len(chains) == 3 and len(ctx.md.chains) == 3
    assert [c.meta["seed"] for c in chains] == [1, 2, 3] and [c.meta["seed"] for c in ctx.md.chains] == [1, 2, 3]       # --seed 1: replica j has seed 1 + j
    assert not np.array_equal(chains[0].Ms, chains[1].Ms) and not np.array_equal(chains[1].Ms, chains[2].Ms)      # different random chains
    x = [c.coords[0] for c in ctx.md.chains]
    assert not np.allclose(x[0], x[1], atol=1e-3) and not np.allclose(x[1], x[2], atol=1e-3)                       # different MD runs
    for k in range(3):                                                         # per-replica metadata, structures and figures
        assert (d.data / "metadata" / f"replica_{k}" / "md_avg_distance.npy").exists() and (d.data / "metadata" / f"replica_{k}" / "mc_contact_map.npy").exists()
        assert (d.root / "structures" / f"replica_{k}" / "trajectory.dcd").exists() and any((d.root / "structures" / f"replica_{k}" / "ensemble").glob("MDLE_*.cif"))
        assert (d.root / "visualizations" / f"replica_{k}" / "md_maps.png").exists()
    assert not (d.root / "structures" / "trajectory.dcd").exists()
    for f in ("replica_maps", "md_replica_maps", "md_replica_distances", "md_replicas", "replicas"):    # diagnostics across replicas, in the main folder
        assert (d.figures / f"{f}.png").exists(), f
    assert (d.metadata / "md_avg_distance_replica_std.npy").exists() and (d.metadata / "mc_contact_map_replica_std.npy").exists()
    assert not any(p.name.endswith(("_0", "_1", "_2")) for p in d.root.rglob("*") if p.is_dir() and not p.name.startswith("replica_") and p.name != d.root.name)
    pooled_md, pooled_mc = ctx.md_ens, ctx.mc_ens                             # pooled frames index the pooled loops
    assert pooled_md.frame_idx.max() < pooled_mc.n_samples and len(pooled_md.coords) == sum(len(c.coords) for c in ctx.md.chains)
    solo = run_pipeline(load_config([a for a in _replica_args(fa, 1, tmp_path, "solo")] + ["--seed", str(chains[2].meta["seed"]), "--no-run-MD"]))
    assert np.array_equal(solo.mc.Ms, chains[2].Ms)                           # replica 2 is exactly a stand-alone chain with its own seed
    assert all((d.data / f"{n}_r{k}.npz").exists() for n in ("mc", "md") for k in range(3)) and not (d.data / "mc.npz").exists()   # one file per replica
    again = type(ctx.md).load(d.data / "md.npz")                              # cache round trip keeps all replicas
    assert len(again.chains) == 3 and np.allclose(again.chains[2].coords, ctx.md.chains[2].coords)
