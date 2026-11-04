"""Read / write sectioned config.ini files.

Reading is forgiving: keys are matched by name in any section, so old flat
'[Main]' configs keep working.
"""
from configparser import ConfigParser

from logger import get_logger
from settings.schema import BY_NAME, PARAMS, SECTIONS

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
