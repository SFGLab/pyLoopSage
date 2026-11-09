"""Convert raw strings (ini / CLI) into typed values.

- Read / write sectioned config.ini files.
- Sanity checks on the final configuration.
- Command-line interface built from the parameter schema.
- Merge defaults < config.ini < command line into one Config object.
"""
import argparse
import re
import shlex
import sys
from configparser import ConfigParser
from pathlib import Path
from types import SimpleNamespace

from loopsage.logger import get_logger
from loopsage.settings.schema import BY_NAME, DEPRECATED, PARAMS, SECTIONS


# ---- Convert raw strings (ini / CLI) into typed values ---------------------

# old key -> (new key, converter of the old value): old config files keep working
_RENAMED = {"contrastive_binding": ("binding_mode", lambda v: "contrastive" if str(v).strip().lower() in _TRUE else "preferential")}

_NUM = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)\s*$")
_TRUE, _FALSE = {"true", "1", "y", "yes", "on"}, {"false", "0", "n", "no", "off"}


def _number(value, unit_expected):
    """Parse '100', '100 femtosecond', '0.1 ps' -> float in the expected unit."""
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUM.match(str(value))
    if not m:
        raise ValueError(f"cannot read a number from {value!r}")
    x, unit = float(m.group(1)), m.group(2).lower()
    if unit_expected == "fs" and unit in ("ps", "picosecond"):
        x *= 1000.0
    return x


def _items(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [s.strip() for s in str(value).strip("[]").split(",") if s.strip()]


def coerce(param, value):
    """Typed value for a Param; None for empty input."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    k = param.kind
    if k == "bool":
        s = str(value).strip().lower()
        if s in _TRUE: return True
        if s in _FALSE: return False
        raise ValueError(f"{param.name}: cannot read a boolean from {value!r}")
    if k == "str":
        return str(value).strip()
    if k == "int":
        return int(round(_number(value, param.unit)))
    if k == "float":
        return _number(value, param.unit)
    if k == "floats":
        return [float(x) for x in _items(value)]
    if k == "strs":
        return [str(x) for x in _items(value)]
    raise ValueError(f"unknown kind {k}")


# ---- Read / write sectioned config.ini files -------------------------------

log = get_logger(__name__)


def read_ini(path) -> dict:
    """Flat {key: raw string} from every section of the file."""
    cp = ConfigParser(interpolation=None, inline_comment_prefixes=(";",), allow_no_value=True)
    if not cp.read(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    out = {}
    for sec in cp.sections():
        for key, val in cp.items(sec):
            known = BY_NAME.get(key)
            if known and sec.lower() not in (known.section, "main"):
                log.debug("'%s' found in [%s]; expected [%s]", key, sec, known.section)
            out[key.lower()] = val
    return out


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def write_ini(path, values: dict, source: dict) -> None:
    """Write the fully resolved configuration, grouped by section, with help comments."""
    lines = ["# LoopSage: resolved configuration (defaults + config file + command line)\n"]
    for sec, title in SECTIONS.items():
        lines.append(f"\n[{sec}]  ; {title}\n")
        for p in (q for q in PARAMS if q.section == sec):
            unit = f" [{p.unit}]" if p.unit else ""
            lines.append(f"; {p.help}{unit}  (from {source.get(p.name, 'default')})\n")
            lines.append(f"{p.name} = {_to_text(values.get(p.name))}\n")
    with open(path, "w") as f:
        f.writelines(lines)


# ---- Sanity checks on the final configuration ------------------------------

def validate(cfg, need_file: bool = True) -> None:
    """Raise ValueError with a clear message for the first problem found (need_file=False: loops come from Python variables)."""
    if need_file and not cfg.interaction_file:
        raise ValueError("an input file is required (-f FILE or interaction_file in config.ini)")
    if cfg.interaction_file and not Path(cfg.interaction_file).is_file():
        raise ValueError(f"input file not found: {cfg.interaction_file}")
    if cfg.interaction_file and Path(cfg.interaction_file).suffix.lower() not in (".bedpe", ".bed", ".narrowpeak"):
        raise ValueError("unsupported input format (use .bedpe, .bed or .narrowPeak)")
    if not cfg.chrom:
        raise ValueError("a chromosome is required (-r CHR:START-END or chrom in config.ini)")
    if cfg.mc_step < 1 or cfg.n_steps < cfg.mc_step:
        raise ValueError("need 1 <= sample-every <= mc-steps")
    if cfg.burnin >= cfg.n_steps:
        raise ValueError(f"burn-in ({cfg.burnin}) must be smaller than mc-steps ({cfg.n_steps})")
    if not 0.0 <= cfg.topo_keep <= 1.0:
        raise ValueError("topo_keep is a probability: use a value between 0 and 1")
    if not 0.0 < cfg.topo_window <= 0.5:
        raise ValueError("topo_window is a share of the frame's MD steps: use a value in (0, 0.5]")
    if cfg.comp_bw_file and cfg.comp_bed_file:
        raise ValueError("give only one compartment file (BigWig or BED)")
    if not 0 <= cfg.verbosity <= 3:
        raise ValueError("verbosity must be between 0 and 3")
    if cfg.motif_pad < 0:
        raise ValueError("motif_pad must be >= 0")
    if cfg.opt_patience < 1 or cfg.opt_min_gain < 0:
        raise ValueError("opt_patience must be >= 1 and opt_min_gain >= 0")
    if cfg.corr_smooth < 0 or cfg.track_smooth < 0:
        raise ValueError("corr_smooth and track_smooth must be >= 0")
    if cfg.min_pet < 0:
        raise ValueError("min_pet must be >= 0")
    if cfg.motif_genome and not Path(cfg.motif_genome).is_file():
        raise ValueError(f"reference genome not found: {cfg.motif_genome}")
    if cfg.motif_pfm and not Path(cfg.motif_pfm).is_file():
        raise ValueError(f"motif file not found: {cfg.motif_pfm}")
    if not 0 < cfg.motif_pvalue < 1:
        raise ValueError("motif_pvalue must be between 0 and 1")
    if (cfg.min_pet or cfg.motif_genome) and Path(cfg.interaction_file).suffix.lower() != ".bedpe":
        raise ValueError("min_pet and motif_genome apply to .bedpe input only")
    if need_file and cfg.bw_strengths and len(cfg.bw_strengths) != len(cfg.bw_files):
        raise ValueError("bw_strengths must have one value per bw_files entry")


# ---- Command-line interface built from the parameter schema ----------------

_EPILOG = """\
examples:
  loopsage -f loops.bedpe -r chr1:75000000-95000000 -n 1000 -o out
  loopsage -c config.ini                 # sectioned config file
  loopsage -c config.ini --no-run-MD -vv # file + overrides, MC only, info messages

verbosity: none = main steps + progress bars | -v + tables | -vv + info messages | -vvv + debug | -q = no console output
Anything not listed here can be set in config.ini, or one-off with --set KEY=VALUE.
"""


_METAVAR = {"interaction_file": "FILE", "out_path": "DIR", "platform": "NAME", "motif_genome": "FASTA", "motif_pfm": "FILE"}  # others: by kind


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="loopsage", description="LoopSage: loop extrusion + 3D chromatin structure pipeline.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOG)
    ap.add_argument("-c", "--config", metavar="FILE", help="sectioned config .ini (flags override it)")
    ap.add_argument("-r", "--region", metavar="CHR:START-END", help="genomic region in bp, e.g. chr1:75000000-95000000")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v tables, -vv info, -vvv debug")
    ap.add_argument("-s", "--set", action="append", metavar="KEY=VALUE", help="set any config parameter (repeatable)")
    groups = {s: ap.add_argument_group(title) for s, title in SECTIONS.items()}
    for p in PARAMS:
        if not p.flag:
            continue
        names = [p.short, p.flag] if p.short else [p.flag]
        unit = f" [{p.unit}]" if p.unit else ""
        text = f"{p.help}{unit} (default: {p.default})"
        kw = dict(dest=p.name, default=argparse.SUPPRESS, help=text)
        if p.kind == "bool":
            groups[p.section].add_argument(*names, action=argparse.BooleanOptionalAction, **kw)
        else:
            meta = _METAVAR.get(p.name, p.kind.upper())
            groups[p.section].add_argument(*names, type=str, choices=p.choices or None,
                                           metavar=None if p.choices else meta, **kw)
    return ap


def parse_region(text: str) -> dict:
    """'chr1:75,000,000-95,000,000' or 'chr1' -> chrom/region_start/region_end."""
    m = re.match(r"^(\w+)(?::([\d,]+)-([\d,]+))?$", text.strip())
    if not m:
        raise ValueError(f"bad region {text!r}; expected CHR:START-END")
    out = {"chrom": m.group(1)}
    if m.group(2):
        out["region_start"], out["region_end"] = m.group(2).replace(",", ""), m.group(3).replace(",", "")
    return out


# ---- Merge defaults < config.ini < command line into one Config object -----

class Config(SimpleNamespace):
    """Attribute access to every parameter; .source says where each came from."""

    def as_dict(self) -> dict:
        return {p.name: getattr(self, p.name) for p in PARAMS}


def _apply(values, source, raw, origin, notes):
    for raw_key, val in raw.items():
        key = raw_key.lower()
        if key in _RENAMED:
            key, conv = _RENAMED[key]
            notes.append(f"'{raw_key.lower()}' was replaced by '{key}'.")
            val = conv(val)
        if key in DEPRECATED:
            notes.append(f"'{key}' is no longer used and was ignored.")
        elif key not in BY_NAME:
            if origin == "cli":
                raise ValueError(f"unknown parameter '{key}'")
            notes.append(f"unknown key '{key}' in config file was ignored.")
        else:
            values[key], source[key] = coerce(BY_NAME[key], val), origin
            if values[key] is None:          # empty value: fall back to default
                values[key], source[key] = BY_NAME[key].default, "default"


def _build(cli: dict, ini_path, sets, verbose: int, command: str, need_file: bool = True) -> Config:
    """Defaults < ini file < explicit values (raises ValueError / FileNotFoundError)."""
    values = {p.name: p.default for p in PARAMS}
    source = {p.name: "default" for p in PARAMS}
    notes = []
    if ini_path:
        _apply(values, source, read_ini(ini_path), "ini", notes)
    cli = dict(cli)
    if cli.get("region"):
        cli.update(parse_region(cli.pop("region")))
    cli.pop("region", None)
    for item in sets or []:
        key, _, val = item.partition("=")
        cli[key.strip()] = val
    _apply(values, source, cli, "cli", notes)
    cfg = Config(**values)
    cfg.source, cfg.notes, cfg.config_file = source, notes, ini_path
    cfg.verbosity = verbose or int(values["verbosity"] or 0)           # -v wins over the ini
    cfg.command = command
    validate(cfg, need_file)
    return cfg


def load_config(argv=None, default_config: bool = False) -> Config:
    """Parse argv; with default_config, fall back to ./config.ini when no input is given."""
    parser = build_parser()
    ns = parser.parse_args(argv)
    if argv is None and len(sys.argv) == 1 and not default_config:
        parser.print_help()
        sys.exit(0)
    ini_path = ns.config
    if ini_path is None and default_config and not getattr(ns, "interaction_file", None) and Path("config.ini").exists():
        ini_path = "config.ini"
    cli = {k: v for k, v in vars(ns).items() if k in BY_NAME}
    cli["region"] = ns.region
    command = "loopsage " + " ".join(shlex.quote(a) for a in (sys.argv[1:] if argv is None else argv))
    try:
        return _build(cli, ini_path, ns.set, ns.verbose, command)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))


def make_config(config=None, region=None, verbose: int = 0, **params) -> Config:
    """Config from Python: `config` is an .ini path, a Config or a dict; `params` are parameters by name and win over it.

    Errors raise ValueError (no usage message, no exit), so a script can catch them.
    """
    if isinstance(config, Config):
        ini, base = config.config_file, config.as_dict()
    elif isinstance(config, dict):
        ini, base = None, dict(config)
    else:
        ini, base = (str(config) if config else None), {}
    region = region or base.pop("region", None)                      # a dict may carry the region too
    cli = {**base, **params}
    shown = " ".join(f"--set {k}={shlex.quote(str(v))}" for k, v in params.items())
    command = " ".join(x for x in ("loopsage", f"-c {shlex.quote(ini)}" if ini else "", f"-r {region}" if region else "", shown) if x)
    cli["region"] = region
    return _build(cli, ini, None, verbose, command, need_file=False)      # loops may come from Python variables
