"""Scan file (.ini) -> `ScanSpec`: the fixed system, the base parameters, the swept axes and the run settings.

[scan]    name, design (grid | oat | random | sobol), replicates, samples, seed, snapshots, burnin_fraction, metrics, md ...
[system]  the sandbox polymer: n_beads, bp_per_bead, n_ctcf, layout, orientation, strength, compartments ...
[base]    fixed loopsage parameters (any key of config.ini)
[sweep]   one line per axis: a list `1, 2, 4`, a range `2:20:2`, `lin(a, b, n)`, `log(a, b, n)` or `log(a, b)`
          (keys of [system] are written `system.n_ctcf`; `T` sets both temperatures)
"""
import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loopsage.settings.loader import coerce
from loopsage.settings.schema import BY_NAME

DESIGNS = ("grid", "oat", "random", "sobol")
ALIASES = {"t": ("t_init", "t_final"), "temperature": ("t_init", "t_final")}            # one key sets several parameters
SYSTEM = {   # key -> (kind, default, help)
    "n_beads": ("int", 300, "polymer length in beads (200-500 is typical)"),
    "bp_per_bead": ("int", 10000, "genomic size of a bead"),
    "n_ctcf": ("int", 12, "number of CTCF sites"),
    "ctcf_spacing": ("int", 0, "beads between sites (0 = use n_ctcf); keeps the density fixed when n_beads is scanned"),
    "layout": ("str", "even", "even | random: where the sites sit"),
    "orientation": ("str", "convergent", "convergent (F R F R ...) | random | forward | reverse"),
    "strength": ("float", 10.0, "binding strength of a site"),
    "strength_spread": ("float", 0.0, "relative spread (log-normal) of the strengths"),
    "compartments": ("str", "none", "none | random: A/B blocks"),
    "comp_block_mb": ("float", 0.6, "mean A/B block length (random compartments)"),
    "bound_radius": ("int", 1, "a leg within this many beads of a site counts as bound"),
    "seed": ("int", 1, "seed of the random layout"),
}
SCAN = {     # key -> (kind, default)
    "name": ("str", "scan"), "design": ("str", "grid"), "replicates": ("int", 4), "samples": ("int", 64), "seed": ("int", 1),
    "snapshots": ("int", 200), "burnin_fraction": ("float", 0.2), "metrics": ("strs", ["default"]), "workers": ("int", 0),
    "md": ("bool", False), "md_frames": ("int", 5), "md_steps": ("int", 2000), "md_platform": ("str", "auto"),
    "topology": ("bool", False), "max_runs": ("int", 5000), "cache": ("bool", True), "metrics_file": ("str", ""),
}
BASE_DEFAULTS = dict(method="Metropolis", n_steps=100_000, t_init=5.0, t_final=5.0, bind_coeff=0.05)   # the scan's own defaults: a lively regime for a handful of sites


@dataclass
class Axis:
    name: str                 # 'n_lef', 'system.n_ctcf', 't_init' ...
    kind: str                 # int | float | str | bool
    values: list = None       # explicit values (grid) or the choices (random designs)
    lo: float = None          # range ends, for log/lin ranges
    hi: float = None
    log: bool = False
    n: int = None             # points of a range on a grid
    base: object = None       # baseline value (oat design)

    @property
    def numeric(self) -> bool:
        return self.kind in ("int", "float")

    @property
    def is_range(self) -> bool:
        return self.lo is not None

    @property
    def label(self) -> str:
        return "T" if self.name in ALIASES else self.name.replace("system.", "")

    def grid(self) -> list:
        """The values this axis takes on a grid / one-at-a-time design."""
        if not self.is_range:
            return list(self.values)
        pts = np.geomspace(self.lo, self.hi, self.n) if self.log else np.linspace(self.lo, self.hi, self.n)
        return self.cast(pts)

    def cast(self, x):
        """Round to int for integer parameters; unique and sorted for lists."""
        if self.kind == "int":
            return sorted({int(round(v)) for v in np.atleast_1d(x)}) if np.ndim(x) else int(round(x))
        return [float(v) for v in x] if np.ndim(x) else float(x)

    def from_unit(self, u):
        """Map u in [0, 1) to a value (random designs): range -> continuous, list -> one of the choices."""
        if self.is_range:
            v = np.exp(np.log(self.lo) + u * (np.log(self.hi) - np.log(self.lo))) if self.log else self.lo + u * (self.hi - self.lo)
            return self.cast(v)
        vals = self.values
        return vals[min(int(u * len(vals)), len(vals) - 1)]


@dataclass
class ScanSpec:
    name: str = "scan"
    design: str = "grid"
    replicates: int = 4
    samples: int = 64
    seed: int = 1
    snapshots: int = 200
    burnin_fraction: float = 0.2
    metrics: list = field(default_factory=lambda: ["default"])
    workers: int = 0
    md: bool = False
    topology: bool = False
    md_frames: int = 5
    md_steps: int = 2000
    md_platform: str = "auto"
    max_runs: int = 5000
    cache: bool = True
    metrics_file: str = ""
    system: dict = field(default_factory=dict)
    base: dict = field(default_factory=dict)
    axes: list = field(default_factory=list)
    plots: dict = field(default_factory=dict)
    source: str = ""          # the text of the scan file, for the report
    warnings: list = field(default_factory=list)
    command: str = ""

    def to_ini(self) -> str:
        return self.source


# ---- Parse values ---------------------------------------------------------------------------------

_FUNC = re.compile(r"^(lin|log)\s*\(\s*([^,]+)\s*,\s*([^,)]+?)\s*(?:,\s*(\d+)\s*)?\)$", re.I)
_RANGE = re.compile(r"^\s*([-+\d.eE]+)\s*:\s*([-+\d.eE]+)\s*:\s*([-+\d.eE]+)\s*$")


def _kind(name):
    if name.startswith("system."):
        key = name[7:]
        if key not in SYSTEM:
            raise ValueError(f"unknown system key '{key}' (known: {', '.join(SYSTEM)})")
        return SYSTEM[key][0]
    if name in ALIASES:
        return "float"
    if name not in BY_NAME:
        raise ValueError(f"unknown parameter '{name}' in [sweep]")
    k = BY_NAME[name].kind
    if k not in ("int", "float", "bool", "str"):
        raise ValueError(f"'{name}' cannot be swept (it is a list)")
    return k


def _typed(kind, text):
    text = str(text).strip()
    if kind == "bool":
        if text.lower() in ("true", "1", "yes", "on"): return True
        if text.lower() in ("false", "0", "no", "off"): return False
        raise ValueError(f"cannot read a boolean from {text!r}")
    if kind == "str":
        return text
    return int(round(float(text))) if kind == "int" else float(text)


def parse_axis(name: str, text: str) -> Axis:
    """One [sweep] line -> Axis."""
    name, text = name.strip().lower(), text.strip()
    kind = _kind(name)
    ax = Axis(name, kind)
    m, r = _FUNC.match(text), _RANGE.match(text)
    if m:
        if kind not in ("int", "float"):
            raise ValueError(f"{name}: ranges need a numeric parameter")
        ax.log, ax.lo, ax.hi = m.group(1).lower() == "log", float(m.group(2)), float(m.group(3))
        ax.n = int(m.group(4)) if m.group(4) else 5
        if ax.log and min(ax.lo, ax.hi) <= 0:
            raise ValueError(f"{name}: log ranges need positive ends")
        if ax.lo >= ax.hi or ax.n < 2:
            raise ValueError(f"{name}: a range needs lo < hi and at least 2 points")
    elif r and kind in ("int", "float"):
        a, b, s = (float(x) for x in r.groups())
        if s <= 0 or b < a:
            raise ValueError(f"{name}: bad range {text!r} (start:stop:step)")
        ax.values = [int(round(v)) if kind == "int" else float(v) for v in np.arange(a, b + s / 2, s)]
    else:
        ax.values = [_typed(kind, t) for t in text.split(",") if t.strip()]
    if not ax.is_range and len(ax.values or []) < 2:
        raise ValueError(f"{name}: an axis needs at least two values")
    if ax.numeric and not ax.is_range:
        ax.values = sorted(set(ax.values))
    return ax


def _read(path_or_text):
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None, delimiters=("=",))
    p, raw = Path(str(path_or_text)), str(path_or_text)
    if "\n" not in raw and "[" not in raw and not p.is_file():
        raise FileNotFoundError(f"scan file not found: {raw}")
    text = p.read_text() if len(raw) < 400 and "\n" not in raw and p.is_file() else raw
    cp.read_string(text)
    return cp, text


def load_scan(source, sets=None) -> ScanSpec:
    """Read a scan file (path) or its text; `sets` are overrides like 'scan.replicates=8' or 'sweep.T=1,2,4'. Raises ValueError."""
    cp, text = _read(source)
    for item in sets or []:
        key, _, val = item.partition("=")
        sec, _, name = key.strip().partition(".")
        if not name or sec not in ("scan", "system", "base", "sweep", "plots"):
            raise ValueError(f"bad --set {item!r}; use section.key=value with section scan, system, base, sweep or plots")
        if not cp.has_section(sec):
            cp.add_section(sec)
        cp[sec][name.strip().lower()] = val.strip()
        text += f"\n# override: {sec}.{name.strip()} = {val.strip()}"
    unknown = [s for s in cp.sections() if s not in ("scan", "system", "base", "sweep", "plots")]
    if unknown:
        raise ValueError(f"unknown section [{unknown[0]}] (use scan, system, base, sweep, plots)")
    spec = ScanSpec(source=text)
    for k, v in (cp["scan"].items() if cp.has_section("scan") else []):
        if k not in SCAN:
            raise ValueError(f"unknown key '{k}' in [scan] (known: {', '.join(SCAN)})")
        kind = SCAN[k][0]
        setattr(spec, k, [s.strip() for s in v.split(",") if s.strip()] if kind == "strs" else _typed(kind, v))
    spec.system = {k: d for k, (_, d, _) in SYSTEM.items()}
    for k, v in (cp["system"].items() if cp.has_section("system") else []):
        if k not in SYSTEM:
            raise ValueError(f"unknown key '{k}' in [system] (known: {', '.join(SYSTEM)})")
        spec.system[k] = _typed(SYSTEM[k][0], v)
    spec.base = dict(BASE_DEFAULTS)
    for k, v in (cp["base"].items() if cp.has_section("base") else []):
        for key in ALIASES.get(k, (k,)):
            if key not in BY_NAME:
                raise ValueError(f"unknown parameter '{k}' in [base]")
            spec.base[key] = coerce(BY_NAME[key], v)
    for k, v in (cp["sweep"].items() if cp.has_section("sweep") else []):
        spec.axes.append(parse_axis(k, v))
    spec.plots = dict(cp["plots"]) if cp.has_section("plots") else {}
    check(spec)
    return spec


def check(spec: ScanSpec):
    """Validate the combination of settings."""
    if spec.design not in DESIGNS:
        raise ValueError(f"design must be one of {', '.join(DESIGNS)}")
    if not spec.axes:
        raise ValueError("the [sweep] section needs at least one axis")
    names = [a.name for a in spec.axes]
    if len(set(names)) != len(names):
        raise ValueError("an axis appears twice in [sweep]")
    if spec.replicates < 1 or spec.snapshots < 20:
        raise ValueError("replicates must be >= 1 and snapshots >= 20")
    if spec.design in ("random", "sobol"):
        if len(spec.axes) < 2 and spec.design == "sobol":
            raise ValueError("a sobol design needs at least two axes")
    for a in spec.axes:
        if a.name == "system.n_beads" and any(v < 40 for v in (a.values or [a.lo])):
            raise ValueError("system.n_beads must be at least 40")
        if a.name in ("t_init", "t_final") and spec.base.get("method") == "Annealing":
            spec.warnings.append(f"'{a.name}' is swept with method = Annealing: the temperature changes during the run; use `T` with Metropolis for a clean temperature scan.")
    from loopsage.settings.schema import BY_NAME as _P
    topo_axes = [a.name for a in spec.axes if a.name in _P and (_P[a.name].section == "topology" or a.name.startswith("ev_ff"))]
    if topo_axes and not spec.topology:
        spec.topology = True
        spec.warnings.append(f"'{topo_axes[0]}' is a topology parameter: the 3D and link analysis were switched on (topology = true).")
    if spec.topology and not spec.md:
        spec.md = True
        spec.warnings.append("topology = true needs the 3D step: md was switched on.")
    if spec.base.get("method") not in ("Metropolis", "Annealing"):
        raise ValueError("method must be Metropolis or Annealing")
