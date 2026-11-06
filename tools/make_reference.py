"""Write markdown/reference/cli.md and config.md from the parameter schema (src/settings/schema.py).

Run from the repository root:  PYTHONPATH=src python tools/make_reference.py
"""
from pathlib import Path

from settings.schema import PARAMS, SECTIONS

OUT = Path(__file__).resolve().parents[1] / "markdown" / "reference"
TYPES = dict(int="integer", float="number", bool="true / false", str="text", floats="list of numbers", strs="list of text")


def show(v) -> str:
    """Default value as it is written in a config file."""
    if v is None or v == []:
        return "*(empty)*"
    if isinstance(v, str) and v.endswith("classic_sm_ff.xml"):          # bundled file: show it relative to the repository
        v = "src/molecular_dynamics/forcefields/classic_sm_ff.xml"
    return f"`{', '.join(map(str, v))}`" if isinstance(v, list) else f"`{v}`"


def table(head, rows) -> str:
    """Markdown table."""
    esc = lambda s: str(s).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] +
                     ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]) + "\n"


def flag_text(p) -> str:
    """Command-line spelling of a parameter."""
    names = f"`{p.short}`, `{p.flag}`" if p.short else f"`{p.flag}`"
    return names + (f" / `--no-{p.flag[2:]}`" if p.kind == "bool" else "")


def cli_page() -> str:
    parts = [CLI_HEAD]
    for sec, title in SECTIONS.items():
        rows = [[flag_text(p), TYPES[p.kind] + (f" ({p.unit})" if p.unit else ""), show(p.default), p.help]
                for p in PARAMS if p.section == sec and p.flag]
        if rows:
            parts.append(f"### {title}\n\n" + table(["Option", "Type (unit)", "Default", "What it does"], rows))
    return "\n".join(parts) + CLI_TAIL


def config_page() -> str:
    parts = [CONFIG_HEAD]
    for sec, title in SECTIONS.items():
        rows = [[f"`{p.name}`<br>" + (f"`{p.flag}`" if p.flag else "*ini only*"), TYPES[p.kind] + (f" ({p.unit})" if p.unit else ""), show(p.default), p.help]
                for p in PARAMS if p.section == sec]
        parts.append(f"## `[{sec}]`: {title}\n\n" + table(["Key / CLI option", "Type (unit)", "Default", "What it does"], rows))
    return "\n".join(parts)


CLI_HEAD = """# Command-line interface

LoopSage is run with one command. Every option below can also be written in a [config file](config.md); the command line always wins.

```bash
loopsage [options]
```

!!! tip "Where options are read from"
    From weakest to strongest: built-in defaults, then the config file (`-c`), then command-line options, then `-s KEY=VALUE`.
    So a config file holds your usual settings and the command line changes one thing at a time.

## Examples

```bash
# Minimal run: a loop file and a region
loopsage -f loops.bedpe -r chr1:77000000-95000000

# Everything from a config file, but without the 3D stage and with info messages
loopsage -c config.ini --no-run-MD -vv

# Read CTCF orientation from the genome, keep loops with at least 3 PET counts, 4 threads
loopsage -f loops.bedpe -r chr1:77000000-95000000 --motif-genome hg38.fa --min-pet 3 -t 4

# Set any config key that has no option of its own
loopsage -c config.ini -s epi_coeff=0.5 -s bind_coeff=2
```

## General options

| Option | What it does |
|---|---|
| `-h`, `--help` | Show the help text and exit. |
| `-c FILE`, `--config FILE` | Read settings from a sectioned `.ini` file (see [Config files](config.md)). |
| `-r CHR:START-END`, `--region CHR:START-END` | Genomic region in base pairs, for example `chr1:75000000-95000000`. |
| `-v`, `-vv`, `-vvv` | Console detail. None: main steps and progress bars. `-v`: plus tables. `-vv`: plus info messages. `-vvv`: plus debug. |
| `-s KEY=VALUE`, `--set KEY=VALUE` | Set any config key from the command line. Can be repeated. |

## Options by topic

Switches that are on or off come in pairs: `--run-MD` turns a step on and `--no-run-MD` turns it off. Lists in the *Type* column show the unit in brackets when the value has one.
"""

CLI_TAIL = """
## Output

Every run gets its own folder, `results/<chrom>_<start>_<end>_<run>/`, with the report in `report/report.html`, the figures in `visualizations/`, the 3D structures in `structures/`, and the numerical results in `data/` (matrices in `data/metadata/` as `.npy` files).
Use `-o` to choose another results folder and `--run N` to fix the run number.

## Run only some stages

The pipeline has seven stages: reading the input, Monte Carlo, molecular dynamics, 3D structure analysis, validation, linking analysis and the report.
Each stage can also be started alone with the same options; it re-uses the saved result of the earlier stages from the latest run.

```bash
python src/pipeline/stage_mc.py -c config.ini      # Monte Carlo (and the input before it)
python src/pipeline/stage_md.py -c config.ini      # up to the 3D molecular dynamics
```
"""

CONFIG_HEAD = """# Config files

A config file keeps all the settings of a run in one place, so a run can be repeated exactly.
Pass it with `-c`:

```bash
loopsage -c config.ini
```

`config_example.ini` in the repository root lists every key with its default and a one-line description. Copy it and change what you need.

## How to write one

A config file is a plain text file in the INI format: *sections* in square brackets, and one `key = value` per line.

```ini
[input]
interaction_file = /data/loops.bedpe   ; the loops to model
chrom = chr1
region_start = 77000000                ; in base pairs
region_end = 95000000

[monte_carlo]
mc_steps = 20000
method = Annealing

[molecular_dynamics]
run_md = True
```

Rules:

- **Sections** group the keys by topic (tables below). A key must be written in its own section.
- **Comments** are whole lines starting with `#` or `;`. A comment may also follow a value on the same line, after `;`.
- **Empty values** (`seed =`) mean "not set": LoopSage then picks a sensible automatic value.
- **Booleans** are written `True` / `False` (also `yes` / `no`, `1` / `0`).
- **Lists** are comma-separated: `bw_files = a.bw, b.bw`.
- **Units** may follow a number (`step_size = 100 fs`); the unit of each key is given in the tables.
- **Only the keys you change are needed.** Anything missing takes its default.
- **Every command-line option is also a config key**, but not the other way round: the config file has more (advanced) keys, marked *ini only* below.

!!! tip "Check what was used"
    Each run saves the full list of settings it used in `config_resolved.ini` inside the run folder, so any run can be reproduced from its own folder.

## All keys

**Type** tells how to write the value. Under each key is the equivalent command-line option, or *ini only* when the key has none.
"""

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cli.md").write_text(cli_page(), encoding="utf-8")
    (OUT / "config.md").write_text(config_page(), encoding="utf-8")
    print("wrote", OUT)
