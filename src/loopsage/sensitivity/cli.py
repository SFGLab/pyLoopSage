"""The `loopsage-scan` command: sensitivity scans of sandboxed loop-extrusion simulations, combined into one report."""
import argparse
import shlex
import sys
from pathlib import Path

TEMPLATE = """# loopsage-scan: a sensitivity scan. Run it with:  loopsage-scan scan.ini
# Every run is a small simulation of a synthetic polymer; the scan writes one combined report.

[scan]
name       = temperature_and_rings
design     = grid          # grid | oat (one at a time) | random | sobol
replicates = 4             # independent seeds per point (the same seeds at every point)
snapshots  = 200           # saved snapshots per run
metrics    = default       # default | all | a list, e.g. mean_loop_length, bound_fraction
# md       = true          # also build 3D structures (slow: use few points)
# topology = true          # link and crossing analysis of the 3D structures (needs md)

[system]                   # the synthetic polymer (200-500 beads is typical)
n_beads     = 300
bp_per_bead = 10000
n_ctcf      = 12
layout      = even         # even | random
orientation = convergent   # convergent | random | forward | reverse

[base]                     # fixed loopsage parameters (any key of config.ini)
n_steps = 100000
method  = Metropolis

[sweep]                    # what to scan: a list, start:stop:step, lin(a, b, n) or log(a, b, n)
T     = lin(1, 16, 12)     # temperature (sets t_init and t_final); 10+ values along one input reveal phase transitions
n_lef = 4, 12              # number of loop extrusion factors
"""


def build_parser():
    p = argparse.ArgumentParser(prog="loopsage-scan", description="Sensitivity scans of sandboxed loop-extrusion simulations, with one combined report (sensitivity_report.html).")
    p.add_argument("scan", nargs="?", help="scan file (.ini)")
    p.add_argument("-o", "--out", help="output folder (default: results/scan_<name>)")
    p.add_argument("-j", "--workers", type=int, default=None, help="worker processes (default: all cores)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a setting of the scan file, e.g. --set scan.replicates=8 --set sweep.T=1,2,4")
    p.add_argument("--md", action="store_true", help="also run the 3D molecular dynamics and report structure metrics")
    p.add_argument("--topology", action="store_true", help="also analyse links and chain crossings in 3D (implies --md)")
    p.add_argument("--dry-run", action="store_true", help="count the runs, time one of them and print the estimated total time; do not run the scan")
    p.add_argument("--replot", action="store_true", help="rebuild the report from the cache without running anything")
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write the cache of finished runs")
    p.add_argument("--init", metavar="FILE", help="write a template scan file and exit")
    p.add_argument("--list-metrics", action="store_true", help="list the available metrics and exit")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    return p


def _list_metrics():
    from loopsage.sensitivity import metrics as M
    for group, d in (("Monte Carlo", M.METRICS), ("3D structure (md)", M.MD_METRICS), ("Topology (md + topology)", M.TOPO_METRICS)):
        print(f"\n{group}")
        for k, v in d.items():
            lab, help_ = (v.label, v.help) if hasattr(v, "label") else (v[0], v[2])
            print(f"  {k:<24} {lab}" + (f" - {help_}" if help_ else ""))


def cli(argv=None):
    """Entry point; returns None (errors exit with a message)."""
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.init:
        path = Path(ns.init)
        if path.exists():
            parser.error(f"{path} exists; it was not overwritten")
        path.write_text(TEMPLATE)
        print(f"Wrote {path}. Edit it, then run: loopsage-scan {path}")
        return
    if ns.list_metrics:
        return _list_metrics()
    if not ns.scan:
        parser.print_help()
        return
    from loopsage.logger import console, set_quiet
    from loopsage.sensitivity.runner import estimate, run_scan
    from loopsage.sensitivity.spec import check, load_scan
    try:
        spec = load_scan(ns.scan, ns.set)
        spec.command = "loopsage-scan " + " ".join(shlex.quote(a) for a in (sys.argv[1:] if argv is None else argv))
        spec.md = spec.md or ns.md or ns.topology
        spec.topology = spec.topology or ns.topology
        spec.cache = spec.cache and not ns.no_cache
        check(spec)
        out = Path(ns.out) if ns.out else Path.cwd() / "results" / f"scan_{spec.name}"
        if ns.dry_run:
            total, n, one = estimate(spec, ns.workers)
            set_quiet(False)
            console.print(f"[bold]{n} runs[/bold] ({spec.design}, {spec.replicates} replicates). One typical run took {one:.1f} s; "
                          f"estimated total {total:.0f} s (~{total / 60:.1f} min) on {ns.workers or spec.workers or 'all'} worker(s).")
            return
        res = run_scan(spec, out, workers=ns.workers, progress=not ns.quiet, replot_only=ns.replot)
        page = res.report(out / "sensitivity_report.html")
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        parser.error(str(e))
    set_quiet(False)                                       # the in-process runs silence the console
    console.print(f"[bold]{len(res.tasks)} runs[/bold] in {res.wall_s:.0f} s ({res.cache_hits} from the cache). Report: {page}")
    if res.failures:
        console.print(f"[yellow]{len(res.failures)} run(s) failed; see the quality section of the report.[/yellow]")


if __name__ == "__main__":
    cli()
