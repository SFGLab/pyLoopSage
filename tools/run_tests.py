#!/usr/bin/env python3
"""Run the whole test suite in parallel (one pytest process per test file) and show a report.

    python tools/run_tests.py                 # all tests, workers = half of the cores
    python tools/run_tests.py -j 8            # 8 parallel workers
    python tools/run_tests.py -k contrast     # only tests whose name matches
    python tools/run_tests.py tests/test_src_optimizer.py tests/test_src_hic.py
    python tools/run_tests.py --html          # also write test_report/report.html

Needs pytest and rich. Exit code 0 only if every test passed.
"""
import argparse
import html
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich import box
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
console = Console()


class State:
    """What the live view shows for one test file."""
    def __init__(self, path: Path):
        self.name = path.stem.removeprefix("test_")
        self.status, self.pct, self.current, self.start, self.end, self.done, self.bad = "waiting", 0, "", 0.0, 0.0, 0, 0


_LINE = re.compile(r"::(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s*(?:\[\s*(\d+)%\])?")


def run_file(path: Path, extra: list, tmp: Path, st: State) -> dict:
    """Run one test file in its own pytest process (streaming its -v output into `st`); returns counts, time, failures and slowest tests."""
    xml = tmp / f"{path.stem}.xml"
    cmd = [sys.executable, "-m", "pytest", str(path), "-v", "-p", "no:cacheprovider", f"--junitxml={xml}", "--tb=short", *extra]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    t0 = st.start = time.time()
    st.status = "running"
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0)
    buf, chunks = "", []
    while True:
        data = os.read(proc.stdout.fileno(), 4096).decode(errors="replace")
        if not data:
            break
        chunks.append(data)
        buf += data
        *lines, buf = buf.split("\n")
        for line in lines:
            m = _LINE.search(line)
            if m:
                st.done += 1
                st.bad += m.group(2) in ("FAILED", "ERROR")
                st.pct = int(m.group(3)) if m.group(3) else st.pct
        tail = re.search(r"::(\S+)\s*$", buf)                      # a test name without a result yet: this one is running
        st.current = tail.group(1) if tail else st.current
    proc.wait()
    st.end, st.status = time.time(), "failed" if proc.returncode not in (0, 5) else "done"
    st.current, st.pct = "", 100
    proc.log = "".join(chunks)
    out = dict(name=st.name, passed=0, failed=0, skipped=0, seconds=time.time() - t0,
               failures=[], slow=[], log=proc.log, code=proc.returncode)
    if xml.exists():
        for case in ET.parse(xml).getroot().iter("testcase"):
            name, dt = case.get("name"), float(case.get("time", 0))
            bad = case.find("failure") if case.find("failure") is not None else case.find("error")
            if bad is not None:
                out["failed"] += 1
                out["failures"].append((f"{path.stem}::{name}", (bad.get("message") or "")[:200], bad.text or ""))
            elif case.find("skipped") is not None:
                out["skipped"] += 1
            else:
                out["passed"] += 1
            out["slow"].append((dt, f"{path.stem}::{name}"))
    elif proc.returncode not in (0, 5):                           # pytest crashed before writing a report (e.g. an import error)
        out["failed"] = 1
        out["failures"].append((path.stem, "pytest did not finish", out["log"][-1500:]))
    return out


def live_view(states: list, t0: float) -> Table:
    """Every test file with its state: waiting (dim), running (spinner, current test, progress), finished (counts)."""
    t = Table(title=f"{sum(x.status in ('done', 'failed') for x in states)}/{len(states)} files finished, {time.time() - t0:.0f} s", box=box.SIMPLE_HEAD,
              header_style="bold", title_justify="left")
    for col, just in (("", "left"), ("Module", "left"), ("Progress", "left"), ("Now running", "left"), ("Time (s)", "right")):
        t.add_column(col, justify=just, no_wrap=True, overflow="ellipsis")
    for x in states:
        if x.status == "waiting":
            t.add_row(Text("·", style="dim"), Text(x.name, style="dim"), Text("waiting", style="dim"), "", "")
        elif x.status == "running":
            bar = "█" * (x.pct // 10) + "░" * (10 - x.pct // 10)
            t.add_row(Spinner("dots", style="#6a3de8"), x.name, f"{bar} {x.pct:>3}%  {x.done} tests", Text(x.current[:46], style="#6a3de8"), f"{time.time() - x.start:.0f}")
        else:
            ok = x.status == "done" and not x.bad
            t.add_row(Text("✓" if ok else "✗", style="green" if ok else "bold red"), x.name, f"{x.done} tests" + (f", {x.bad} failed" if x.bad else ""), "", f"{x.end - x.start:.0f}")
    return t


def table(results: list, title: str) -> Table:
    t = Table(title=title, box=box.SIMPLE_HEAD, header_style="bold", title_justify="left")
    for col, just in (("Module", "left"), ("Passed", "right"), ("Failed", "right"), ("Skipped", "right"), ("Time (s)", "right")):
        t.add_column(col, justify=just)
    for r in sorted(results, key=lambda r: r["name"]):
        t.add_row(r["name"], str(r["passed"]), f"[red]{r['failed']}[/red]" if r["failed"] else "-", str(r["skipped"]) if r["skipped"] else "-", f"{r['seconds']:.0f}")
    t.add_section()
    t.add_row("total", str(sum(r["passed"] for r in results)), str(sum(r["failed"] for r in results)) or "-", str(sum(r["skipped"] for r in results)) or "-",
              f"{sum(r['seconds'] for r in results):.0f}", style="bold")
    return t


def write_html(results: list, wall: float, path: Path) -> None:
    """Self-contained white report: summary per module, failures, slowest tests."""
    rows = "".join(f"<tr><td>{html.escape(r['name'])}</td><td>{r['passed']}</td><td class='{'bad' if r['failed'] else ''}'>{r['failed']}</td>"
                   f"<td>{r['skipped']}</td><td>{r['seconds']:.0f}</td></tr>" for r in sorted(results, key=lambda r: r["name"]))
    fails = "".join(f"<h3>{html.escape(n)}</h3><p>{html.escape(m)}</p><pre>{html.escape(d)}</pre>" for r in results for n, m, d in r["failures"])
    slow = "".join(f"<tr><td>{html.escape(n)}</td><td>{s:.1f}</td></tr>" for s, n in sorted((x for r in results for x in r["slow"]), reverse=True)[:40])
    p, f, s = (sum(r[k] for r in results) for k in ("passed", "failed", "skipped"))
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>LoopSage tests</title>
<style>body{{font:15px system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#fff;color:#111}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{padding:.35rem .6rem;border-bottom:1px solid #e3ddf5;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{border-bottom:2px solid #6a3de8}}.bad{{color:#c00;font-weight:600}}
pre{{background:#f4f1fc;padding:.7rem;overflow:auto;font-size:12px}}h1{{margin-bottom:0}}</style>
<h1>LoopSage tests</h1><p>{p} passed, {f} failed, {s} skipped in {wall:.0f} s (wall time). {time.strftime('%Y-%m-%d %H:%M')}</p>
<table><tr><th>Module</th><th>Passed</th><th>Failed</th><th>Skipped</th><th>Time (s)</th></tr>{rows}</table>
{'<h2>Failures</h2>' + fails if fails else '<p><b>All tests passed.</b></p>'}
<h2>Slowest tests</h2><table><tr><th>Test</th><th>Seconds</th></tr>{slow}</table>""")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="test files (default: all in tests/)")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel workers (default: half of the cores)")
    ap.add_argument("-k", dest="expr", default="", help="only tests matching this expression")
    ap.add_argument("--html", action="store_true", help="also write test_report/report.html")
    a = ap.parse_args()
    files = [Path(f).resolve() for f in a.files] or sorted((ROOT / "tests").glob("test_*.py"))
    extra = ["-k", a.expr] if a.expr else []
    console.print(f"[bold]LoopSage tests[/bold]: {len(files)} files, {a.jobs} workers")
    results, t0 = [], time.time()
    states = [State(f) for f in files]
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.jobs) as pool, Live(get_renderable=lambda: live_view(states, t0), console=console, refresh_per_second=10):
        futs = [pool.submit(run_file, f, extra, Path(tmp), st) for f, st in zip(files, states)]
        results = [fut.result() for fut in futs]
    wall = time.time() - t0
    console.print(table(results, f"Finished in {wall:.0f} s"))
    failed = [x for r in results for x in r["failures"]]
    for name, msg, detail in failed:
        console.rule(f"[red]FAILED[/red] {name}", style="red")
        console.print(detail[-1800:], markup=False, highlight=False)
    slow = sorted((x for r in results for x in r["slow"]), reverse=True)[:5]
    console.print("Slowest: " + "; ".join(f"{n.split('::')[1]} ({s:.0f} s)" for s, n in slow), highlight=False, markup=False)
    if failed:                                                      # the last thing on the screen: which tests failed
        t = Table(title="Failed tests", box=box.SIMPLE_HEAD, header_style="bold red", title_justify="left", title_style="bold red")
        t.add_column("Test"); t.add_column("Reason", overflow="fold")
        for name, msg, _ in failed:
            t.add_row(name, msg.replace("\n", " ")[:160])
        console.print(t)
    if a.html:
        out = ROOT / "test_report" / "report.html"
        write_html(results, wall, out)
        console.print(f"HTML report: {out}")
    console.print("[bold green]All tests passed.[/bold green]" if not failed else f"[bold red]{len(failed)} test(s) failed.[/bold red]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
