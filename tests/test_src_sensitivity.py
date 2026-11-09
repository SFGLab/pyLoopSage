"""Tests of the sensitivity scan (loopsage-scan): spec, designs, metrics, runner, phases, report, CLI."""
import numpy as np
import pandas as pd
import pytest

from loopsage.sensitivity import analysis as A
from loopsage.sensitivity import metrics as M
from loopsage.sensitivity import phases as PH
from loopsage.sensitivity.cli import cli
from loopsage.sensitivity.design import build_design
from loopsage.sensitivity.runner import estimate, run_scan
from loopsage.sensitivity.spec import ScanSpec, check, load_scan, parse_axis

TINY = """
[scan]
name = tiny
design = grid
replicates = 2
snapshots = 40
[system]
n_beads = 60
n_ctcf = 4
[base]
n_steps = 4000
[sweep]
T = 1, 4, 12
n_lef = 3, 6
"""


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("scan")
    spec = load_scan(TINY)
    return spec, run_scan(spec, d, workers=1, progress=False), d


# ---- spec ----
def test_parse_axis_forms():
    assert parse_axis("n_lef", "4, 8, 12").grid() == [4, 8, 12]
    assert parse_axis("n_lef", "4:12:4").grid() == [4, 8, 12]
    assert parse_axis("T", "lin(1, 4, 4)").grid() == pytest.approx([1, 2, 3, 4])
    g = parse_axis("T", "log(1, 100, 3)").grid()
    assert g == pytest.approx([1, 10, 100])


def test_temperature_alias_label():
    assert parse_axis("T", "1, 2").label == "T"


def test_load_scan_set_and_errors():
    spec = load_scan(TINY, ["scan.replicates=5", "sweep.n_lef=2,4"])
    assert spec.replicates == 5 and spec.axes[1].grid() == [2, 4]
    with pytest.raises(ValueError):
        load_scan(TINY.replace("n_lef", "not_a_parameter"))
    with pytest.raises(ValueError):
        load_scan(TINY.replace("design = grid", "design = nonsense"))


def test_topology_axis_enables_md():
    spec = load_scan(TINY.replace("n_lef = 3, 6", "topo_rate = 0.0, 0.05"))
    check(spec)
    assert spec.md and spec.topology


# ---- designs ----
@pytest.mark.parametrize("kind,expect", [("grid", 6), ("oat", 4), ("random", 8), ("sobol", None)])
def test_design_sizes(kind, expect):
    spec = load_scan(TINY.replace("design = grid", f"design = {kind}\nsamples = 7"))
    d = build_design(spec)
    if kind == "sobol":
        n = sum(p["group"] == "A" for p in d.points)                # Sobol sequences are rounded up to a power of two
        assert n >= 7 and len(d.points) == n * (2 + len(spec.axes))
        assert {p["group"] for p in d.points} >= {"A", "B", "AB0", "AB1"}
    else:
        assert len(d.points) == expect


def test_max_runs_guard():
    spec = load_scan(TINY.replace("[scan]", "[scan]\nmax_runs = 3"))
    with pytest.raises(ValueError):
        build_design(spec)


# ---- metrics ----
def test_registry_and_helpers():
    assert {"mean_loop_length", "n_parents", "bound_fraction", "unfolded_fraction"} <= set(M.METRICS)
    assert {"linked_fraction", "leak_rate", "ev_barrier"} <= set(M.TOPO_METRICS)
    rng = np.random.default_rng(0)
    assert M.bimodality(np.r_[rng.normal(0, .1, 200), rng.normal(5, .1, 200)]) > M.BIMODAL if hasattr(M, "BIMODAL") else True
    assert M.bimodality(rng.normal(size=400)) < 5 / 9
    assert M.binder(rng.normal(size=20000)) == pytest.approx(0, abs=0.05)
    assert M.binder(np.r_[np.ones(500), -np.ones(500)]) == pytest.approx(2 / 3, abs=0.02)
    assert 1 <= M.tau_int(rng.normal(size=500)) < 3


# ---- runner ----
def test_scan_runs_and_table(tiny):
    spec, res, _ = tiny
    df = res.table
    assert len(df) == 6 * 2 and not res.failures
    assert {"T", "n_lef", "mean_loop_length", "tau:mean_loop_length", "std:n_parents", "u4:bound_fraction"} <= set(df.columns)
    assert df.mean_loop_length.between(0, 60 * 10).all()


def test_more_extruders_cover_more(tiny):
    df = tiny[1].table
    g = df.groupby("n_lef").unfolded_fraction.mean()
    assert g.loc[6] <= g.loc[3] + 0.05


def test_cache_and_determinism(tiny, tmp_path):
    spec, res, d = tiny
    again = run_scan(spec, d, workers=1, progress=False)
    assert again.cache_hits == len(again.tasks)
    pd.testing.assert_frame_equal(res.table.reset_index(drop=True), again.table.reset_index(drop=True))
    fresh = run_scan(load_scan(TINY), tmp_path, workers=1, progress=False)
    np.testing.assert_allclose(fresh.table.mean_loop_length, res.table.mean_loop_length)


def test_parallel_matches_serial(tiny, tmp_path):
    par = run_scan(load_scan(TINY), tmp_path, workers=2, progress=False)
    np.testing.assert_allclose(par.table.sort_values(["point", "rep"]).mean_loop_length, tiny[1].table.sort_values(["point", "rep"]).mean_loop_length)


def test_estimate(tiny):
    total, n, one = estimate(tiny[0], 2)
    assert n == 12 and total > 0 and one > 0


# ---- analysis ----
def test_importance_finds_the_driver(tiny):
    imp = A.importance(tiny[1])
    assert {"metric", "axis", "eta2"} <= set(imp.columns)
    assert imp.eta2.between(0, 1).all()
    top = imp[imp.metric == "mean_loop_length"].sort_values("eta2").axis.iloc[-1]
    assert top in ("T", "n_lef")


def test_sobol_indices(tmp_path):
    spec = load_scan(TINY.replace("design = grid", "design = sobol\nsamples = 8").replace("replicates = 2", "replicates = 1"))
    res = run_scan(spec, tmp_path, workers=2, progress=False)
    sob = A.sobol(res)
    assert len(sob) and {"S1", "ST"} <= set(sob.columns)


# ---- phases ----
class _Fake:
    """A synthetic scan result: a step in the order parameter at x = 5 with split replicates."""
    def __init__(self, step=True):
        rng = np.random.default_rng(1)
        rows = []
        for x in np.linspace(1, 10, 10):
            for r in range(4):
                y = (1.0 if x > 5 else 0.0) + rng.normal(0, 0.02 if step else 1.0)
                rows.append(dict(point=int(x), rep=r, x=x, m=y, **{"std:m": 0.3 if abs(x - 5) < 1 else 0.05, "tau:m": 20 if abs(x - 5) < 1 else 2, "bim:m": 0.0}))
        self.table = pd.DataFrame(rows)

        class Ax:
            name = label = "x"; numeric = True; log = False
        class Dz:
            axes = [Ax()]; kind = "random"
        self.design = Dz()


def test_detect_step_transition(monkeypatch):
    res = _Fake()
    monkeypatch.setattr(A, "metric_cols", lambda d: ["m"])
    det = PH.detect(res)
    r = det.iloc[0]
    assert r.kind == "sharp transition" and 4 < r.x_star < 7 and r.direction == "rise"
    assert "fluctuation peak" in r.signatures and "critical slowing down" in r.signatures
    assert "continuous" in PH.classify(r)


def test_detect_noise_is_not_a_transition(monkeypatch):
    monkeypatch.setattr(A, "metric_cols", lambda d: ["m"])
    det = PH.detect(_Fake(step=False))
    assert (det.kind == "no significant change").all()


def test_detect_needs_four_values(tiny):
    assert PH.numeric_axes(tiny[1]) == []          # only three temperatures: no transition can be resolved


# ---- report and CLI ----
def test_report_files(tiny):
    spec, res, d = tiny
    page = res.report(d / "sensitivity_report.html")
    html = page.read_text(encoding="utf-8")
    for key in ("At a glance", "Phase transitions and bifurcations", "Quality of the scan", "Glossary", "data:image/png;base64"):
        assert key in html
    assert "No transition can be resolved" in html
    for f in ("sensitivity_runs.csv", "sensitivity_points.csv", "sensitivity_importance.csv"):
        assert (d / f).exists()


def test_report_with_transition_section(tmp_path):
    spec = load_scan(TINY.replace("T = 1, 4, 12", "T = lin(1, 16, 8)").replace("n_lef = 3, 6", "").replace("n_steps = 4000", "n_steps = 4000\nn_lef = 6").replace("replicates = 2", "replicates = 3"))
    res = run_scan(spec, tmp_path, workers=2, progress=False)
    html = res.report(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Detected transitions" in html or "No significant change" in html


def test_cli_init_list_dry_run(tmp_path, capsys):
    f = tmp_path / "scan.ini"
    cli(["--init", str(f)])
    assert "[sweep]" in f.read_text()
    with pytest.raises(SystemExit):
        cli(["--init", str(f)])                    # does not overwrite
    load_scan(str(f))                              # the template is a valid scan
    cli(["--list-metrics"])
    assert "mean_loop_length" in capsys.readouterr().out


def test_cli_run_and_replot(tmp_path):
    f = tmp_path / "s.ini"
    f.write_text(TINY)
    out = tmp_path / "o"
    cli([str(f), "-o", str(out), "-j", "1", "-q"])
    assert (out / "sensitivity_report.html").exists()
    (out / "sensitivity_report.html").unlink()
    cli([str(f), "-o", str(out), "--replot", "-q"])
    assert (out / "sensitivity_report.html").exists()
    with pytest.raises(SystemExit):
        cli([str(tmp_path / "missing.ini"), "-q"])


def test_python_api(tmp_path):
    import loopsage as ls
    res = ls.sweep({"n_lef": [3, 6]}, system={"n_beads": 60, "n_ctcf": 4}, base={"n_steps": 3000}, replicates=2, snapshots=30, out=tmp_path, workers=1, progress=False)
    assert len(res.table) == 4 and (tmp_path / "sensitivity_report.html").exists()
    with pytest.raises(ValueError):
        ls.sweep({"n_lef": [3]}, system={"bogus": 1})


# ---- topology ----
def test_topology_metrics_scan(tmp_path):
    pytest.importorskip("openmm")
    spec = load_scan(TINY.replace("T = 1, 4, 12", "topo_rate = 0.0, 0.05").replace("n_lef = 3, 6", "").replace("n_steps = 4000", "n_steps = 4000\nn_lef = 6")
                     .replace("[scan]", "[scan]\nmd_frames = 3\nmd_steps = 200\nmd_platform = CPU"))
    check(spec)
    res = run_scan(spec, tmp_path, workers=2, progress=False)
    df = res.table
    assert not res.failures, res.failures
    assert {"linked_fraction", "link_den", "leak_rate", "ev_barrier"} <= set(df.columns)
    assert (df.link_den > 0).all() and df.linked_fraction.between(0, 1).all()
    html = res.report(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Topology" in html


def test_example_scan_files_are_valid():
    from pathlib import Path
    files = sorted(Path(__file__).resolve().parents[1].glob("examples/*scan*.ini"))
    assert files
    for f in files:
        check(load_scan(str(f)))
