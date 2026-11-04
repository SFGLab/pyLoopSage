"""Command-line interface built from the parameter schema."""
import argparse
import re

from settings.schema import PARAMS, SECTIONS

_EPILOG = """\
examples:
  python run.py -f loops.bedpe -r chr1:75000000-95000000 -n 1000 -o out
  python run.py -c config.ini                 # sectioned config file
  python run.py -c config.ini --no-run-MD -vv # file + overrides, MC only, info messages

verbosity: none = progress bars only | -v + tables | -vv + info messages | -vvv + debug
Anything not listed here can be set in config.ini, or one-off with --set KEY=VALUE.
"""


_METAVAR = {"interaction_file": "FILE", "out_path": "DIR", "platform": "NAME"}  # others: by kind


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run.py", description="LoopSage: loop extrusion + 3D chromatin structure pipeline.",
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
