"""Python entry point of the scans: `loopsage.sweep(...)`."""
from pathlib import Path

from loopsage.sensitivity.spec import SCAN, SYSTEM, Axis, ScanSpec, _typed, check, parse_axis
from loopsage.settings.loader import coerce
from loopsage.settings.schema import BY_NAME
from loopsage.sensitivity.spec import ALIASES, BASE_DEFAULTS


def _axis(name, v) -> Axis:
    """An axis from a list of values, a range string ('2:20:2', 'log(0.3, 6, 5)') or a dict(lo, hi, log, n)."""
    if isinstance(v, dict):
        fn = "log" if v.get("log") else "lin"
        v = f"{fn}({v['lo']}, {v['hi']}" + (f", {v['n']})" if v.get("n") else ")")
    elif isinstance(v, (list, tuple, range)):
        v = ", ".join(str(x) for x in v)
    return parse_axis(name, str(v))


def sweep(axes: dict, system: dict = None, base: dict = None, design: str = "grid", replicates: int = 4, metrics="default",
          out=None, workers: int = None, md: bool = False, report: bool = True, progress: bool = True, **scan):
    """Run a sensitivity scan and return a `ScanResult` (`.table` has one row per run).

        res = ls.sweep({"T": [0.5, 1, 2, 4], "n_lef": "4:24:4"}, system={"n_ctcf": 12, "n_beads": 300}, out="my_scan")

    `axes` maps a parameter (or `system.<key>`) to a list, a range string or a dict(lo, hi, log, n). `system` sets the sandbox,
    `base` the fixed loopsage parameters, `design` is grid | oat | random | sobol. With `out`, the report is written to
    `<out>/sensitivity_report.html`. Extra keywords are [scan] settings (samples, seed, snapshots, burnin_fraction, md_frames, ...).
    """
    from loopsage.sensitivity.runner import run_scan
    spec = ScanSpec(design=design, replicates=replicates, md=md, metrics=[metrics] if isinstance(metrics, str) else list(metrics))
    for k, v in scan.items():
        if k not in SCAN:
            raise ValueError(f"unknown setting '{k}' (known: {', '.join(SCAN)})")
        setattr(spec, k, v)
    spec.system = {k: d for k, (_, d, _) in SYSTEM.items()}
    for k, v in (system or {}).items():
        if k not in SYSTEM:
            raise ValueError(f"unknown system key '{k}' (known: {', '.join(SYSTEM)})")
        spec.system[k] = _typed(SYSTEM[k][0], v)
    spec.base = dict(BASE_DEFAULTS)
    for k, v in (base or {}).items():
        for key in ALIASES.get(k.lower(), (k.lower(),)):
            if key not in BY_NAME:
                raise ValueError(f"unknown parameter '{k}'")
            spec.base[key] = v if not isinstance(v, str) else coerce(BY_NAME[key], v)
    spec.axes = [_axis(k, v) for k, v in axes.items()]
    spec.source = f"# built from Python\n# axes: {axes}\n# system: {system}\n# base: {base}\n# design: {design}, replicates: {replicates}"
    check(spec)
    out = Path(out or "loopsage_scan")
    res = run_scan(spec, out, workers=workers, progress=progress)
    if report:
        res.report(out / "sensitivity_report.html")
    return res
