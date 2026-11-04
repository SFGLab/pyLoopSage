"""Merge defaults < config.ini < command line into one Config object."""
import sys
from pathlib import Path
from types import SimpleNamespace

from settings.cli import build_parser, parse_region
from settings.coerce import coerce
from settings.ini import read_ini
from settings.schema import BY_NAME, DEPRECATED, PARAMS
from settings.validate import validate


class Config(SimpleNamespace):
    """Attribute access to every parameter; .source says where each came from."""

    def as_dict(self) -> dict:
        return {p.name: getattr(self, p.name) for p in PARAMS}


def _apply(values, source, raw, origin, notes):
    for key, val in raw.items():
        key = key.lower()
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


def load_config(argv=None, default_config: bool = False) -> Config:
    """Parse argv; with default_config, fall back to ./config.ini when no input is given."""
    parser = build_parser()
    ns = parser.parse_args(argv)
    if argv is None and len(sys.argv) == 1 and not default_config:
        parser.print_help()
        sys.exit(0)

    values = {p.name: p.default for p in PARAMS}
    source = {p.name: "default" for p in PARAMS}
    notes = []
    ini_path = ns.config
    if ini_path is None and default_config and not getattr(ns, "interaction_file", None) and Path("config.ini").exists():
        ini_path = "config.ini"
    try:
        if ini_path:
            _apply(values, source, read_ini(ini_path), "ini", notes)
        cli = {k: v for k, v in vars(ns).items() if k in BY_NAME}
        if ns.region:
            cli.update(parse_region(ns.region))
        for item in ns.set or []:
            key, _, val = item.partition("=")
            cli[key.strip()] = val
        _apply(values, source, cli, "cli", notes)
        cfg = Config(**values)
        cfg.source, cfg.notes, cfg.verbosity, cfg.config_file = source, notes, ns.verbose, ini_path
        validate(cfg)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    return cfg
