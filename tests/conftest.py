# tests/conftest.py
import sys
from pathlib import Path

import numpy as np
import pytest


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


@pytest.fixture(scope="session")
def demo_bedpe(tmp_path_factory):
    """Synthetic loops on chr1:77-95 Mb (same generator as the demos)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
    from make_demo_data import write_bedpe
    return write_bedpe(tmp_path_factory.mktemp("demo") / "loops.bedpe", n_loops=40)


@pytest.fixture
def fast_args(tiny_args):
    """tiny_args without figures: for tests that check numbers, files or logs, not pictures (a tiny run drops from about 30 s to 3 s)."""
    return tiny_args + ["--no-figures"]


@pytest.fixture(scope="session")
def figure_run(demo_bedpe, tmp_path_factory):
    """One tiny run WITH figures and report, shared by all tests that look at them."""
    from loopsage.pipeline.runner import run_pipeline
    from loopsage.settings.loader import load_config
    out = tmp_path_factory.mktemp("figure_run")
    return run_pipeline(load_config(["-f", str(demo_bedpe), "-r", "chr1:77000000-95000000", "-n", "120", "-o", str(out / "results"),
                                     "--mc-steps", "2000", "--sample-every", "100", "--burnin", "400", "--no-run-MD", "--seed", "1"]))


@pytest.fixture
def tiny_args(demo_bedpe, tmp_path):
    """CLI arguments for a very small, fast run (no MD)."""
    return ["-f", str(demo_bedpe), "-r", "chr1:77000000-95000000", "-n", "120", "-o", str(tmp_path / "results"),
            "--mc-steps", "2000", "--sample-every", "100", "--burnin", "400", "--no-run-MD", "--seed", "1"]


# ---- Terminal output: header and a per-module summary table ----------------

_MODULE_DOC = {}


def pytest_report_header(config):
    return "LoopSage test suite: tests are grouped by module, a summary table follows the run"


def pytest_collection_modifyitems(session, config, items):
    for it in items:
        _MODULE_DOC.setdefault(it.module.__name__, (it.module.__doc__ or "").strip().splitlines()[0] if it.module.__doc__ else "")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Rich table: tests per module with outcome counts and time."""
    from collections import defaultdict
    from rich import box
    from rich.console import Console
    from rich.table import Table
    rows = defaultdict(lambda: dict(passed=0, failed=0, skipped=0, time=0.0))
    for kind, bucket in (("passed", "passed"), ("failed", "failed"), ("error", "failed"), ("skipped", "skipped")):
        for rep in terminalreporter.stats.get(kind, []):
            mod = rep.nodeid.split("::")[0].replace("/", ".").removesuffix(".py").split(".")[-1]
            if rep.when == "call" or kind in ("error", "skipped") or (kind == "failed" and rep.when != "call"):
                rows[mod][bucket] += 1
            rows[mod]["time"] += getattr(rep, "duration", 0.0)
    if not rows:
        return
    t = Table(title="Test summary", box=box.SIMPLE_HEAVY, header_style="bold #6250d6")
    for col, just, w in (("Module", "left", 16), ("What it checks", "left", 0), ("Pass", "right", 5), ("Fail", "right", 5),
                         ("Skip", "right", 5), ("Time", "right", 6)):
        t.add_column(col, justify=just, no_wrap=True, overflow="ellipsis", min_width=w, max_width=(w or None) if w and col == "Module" else None)
    tot = dict(passed=0, failed=0, skipped=0, time=0.0)
    for mod in sorted(rows):
        r = rows[mod]
        t.add_row(mod.removeprefix("test_src_").removeprefix("test_"), _MODULE_DOC.get(mod, "").removeprefix("Tests: ")[:48], f"[green]{r['passed']}[/]",
                  f"[bold red]{r['failed']}[/]" if r["failed"] else "0", str(r["skipped"]) if r["skipped"] else "-", f"{r['time']:.1f}s")
        for k in tot:
            tot[k] += r[k]
    t.add_section()
    t.add_row("[bold]total", "", f"[bold green]{tot['passed']}[/]", f"[bold red]{tot['failed']}[/]" if tot["failed"] else "0",
              str(tot["skipped"]) if tot["skipped"] else "-", f"{tot['time']:.1f}s")
    Console(width=max(terminalreporter._tw.fullwidth, 96)).print(t)
