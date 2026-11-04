"""Collects figures, tables and key facts as stages run; rendered at the end.

- Section introductions and glossary for the report.
- Render the collected figures/tables into one self-contained report.html.
"""
import base64
import datetime
import html
from dataclasses import dataclass, field
from pathlib import Path

from settings.schema import PARAMS, SECTIONS as PARAM_SECTIONS


# ---- Collects figures, tables and key facts as stages run; rendered at the end ----

SECTIONS = {
    "input": "Input data",
    "monte_carlo": "Loop extrusion (Monte Carlo)",
    "molecular_dynamics": "3D structures (molecular dynamics)",
    "validation": "Validation against experiment",
    "structure_3d": "3D structure: diagnostics and viewer",
    "knots": "Topology (linking)",
}


@dataclass
class Fig:
    key: str
    title: str
    caption: str
    explain: str
    section: str
    png: bytes


@dataclass
class Tab:
    title: str
    columns: list
    rows: list
    caption: str
    section: str


@dataclass
class Block:
    """Free HTML card (viewer, videos)."""
    title: str
    html: str
    caption: str
    section: str


@dataclass
class Registry:
    items: list = field(default_factory=list)      # Fig | Tab, in creation order
    facts: list = field(default_factory=list)      # (label, value) headline numbers
    timings: list = field(default_factory=list)    # (stage, seconds)
    fig_dir: Path = None

    def reset(self, fig_dir=None):
        self.items, self.facts, self.timings, self.fig_dir = [], [], [], fig_dir

    def add_table(self, title, columns, rows, caption, section):
        self.items.append(Tab(title, list(columns), rows, caption, section))

    def add_block(self, title, html, caption, section):
        self.items.append(Block(title, html, caption, section))

    def add_fig(self, fig: Fig):
        self.items.append(fig)

    def fact(self, label, value):
        self.facts.append((label, str(value)))

    def of(self, section):
        return [i for i in self.items if i.section == section]


REGISTRY = Registry()


# ---- Section introductions and glossary for the report ---------------------

SECTION_INTRO = {
    "input": "What went into the model: the experimental loops or CTCF peaks, converted to bead resolution "
             "(one <em>simulation bead</em> = a fixed stretch of DNA, its size in bp is given in the tables).",
    "monte_carlo": "A stochastic sampler moves loop extrusion factors (LEFs) along the chromatin fibre and "
                   "accepts moves that lower an energy (Metropolis rule). The result is an ensemble of loop "
                   "configurations consistent with the data. Energies are in arbitrary units (a.u.); "
                   "positions are in beads and converted to bp/kb/Mb.",
    "molecular_dynamics": "Each loop configuration becomes a 3D polymer: beads are connected in a chain, loops "
                          "pull their two anchors together, and OpenMM integrates Langevin dynamics. Lengths "
                          "are in nm, energies in kJ/mol, time in fs/ps/ns.",
    "validation": "How well the simulated contacts reproduce the experimental loop signal. Correlations are "
                  "computed after removing positions where both signals are zero.",
    "structure_3d": "How the polymer looks and behaves in 3D: size, shape, packing, scaling and fractal dimension, plus an interactive viewer and videos. Lengths are in nm. Trajectories are aligned and smoothly interpolated between frames for viewing; diagnostics use the real frames only.",
    "knots": "Topology of the 3D structures: whether loops are linked with each other.",
}

GLOSSARY = [
    ("Bead", "One coarse-grained segment of the chromatin fibre. Its genomic size (bp per bead) is region length / number of beads."),
    ("LEF", "Loop extrusion factor (cohesin or condensin). It holds two legs (m, n) and the chromatin between them forms a loop."),
    ("Anchor", "A bead where a loop starts or ends, typically a CTCF binding site. L / R mark left / right anchors."),
    ("Monte Carlo (MC) step", "One sweep of 2 x (number of LEFs) random proposals. Snapshots are saved every <code>sample-every</code> steps."),
    ("Burn-in", "Initial MC steps discarded because the system has not yet equilibrated."),
    ("Metropolis / Annealing", "Metropolis keeps a constant temperature; Annealing lowers it from T<sub>init</sub> to T<sub>final</sub> to find low-energy states."),
    ("a.u.", "Arbitrary units: the MC energy and temperature have no physical unit; only ratios matter."),
    ("Contact map", "Matrix of how often two beads touch, comparable to a Hi-C map."),
    ("P(s)", "Mean contact probability at genomic separation s; real chromatin shows a steady power-law-like decay."),
    ("R<sub>g</sub>", "Radius of gyration: size of the 3D structure (nm)."),
    ("Pearson r / Spearman &rho;", "Correlation coefficients (&minus;1 to 1) for linear and rank agreement."),
    ("O/E", "Observed / expected: a value divided by the typical value at the same distance."),
    ("Linking number", "Integer counting how often two closed loops wind around each other (0 = unlinked)."),
]


# ---- Render the collected figures/tables into one self-contained report.html ----

esc = html.escape
HTML_DIR = Path(__file__).resolve().parents[1] / "html"   # CSS + page template


def _figure(f: Fig) -> str:
    img = base64.b64encode(f.png).decode()
    return (f'<div class="card"><h3>{esc(f.title)}</h3><figure><img alt="{esc(f.title)}" '
            f'src="data:image/png;base64,{img}"><figcaption>{esc(f.caption)}</figcaption></figure>'
            f'<details open><summary>How to read this figure</summary><p>{f.explain}</p></details></div>')


def _block(b: Block) -> str:
    cap = f'<p class="caption">{esc(b.caption)}</p>' if b.caption else ""
    return f'<div class="card"><h3>{esc(b.title)}</h3>{b.html}{cap}</div>'


def _table_html(columns, rows) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table(t) -> str:
    cap = f'<p class="caption">{esc(t.caption)}</p>' if t.caption else ""
    return f'<div class="card"><h3>{esc(t.title)}</h3>{_table_html(t.columns, t.rows)}{cap}</div>'


def _overview() -> str:
    stats = "".join(f'<div class="stat"><small>{esc(k)}</small><span>{esc(v)}</span></div>' for k, v in REGISTRY.facts)
    timing = _table_html(["Stage", "Wall time"], [[s, t] for s, t in REGISTRY.timings])
    return (f'<section id="overview"><h2>Overview</h2><p class="intro">Headline numbers of this run, and how long each '
            f'stage took. Figures come with a short explanation; unfamiliar terms are defined in the glossary.</p>'
            f'<div class="cards">{stats}</div><div class="card"><h3>Run time per stage</h3>{timing}</div></section>')


def _parameters(cfg) -> str:
    values, tags = cfg.as_dict(), {"default": "", "ini": "ini", "cli": "cli"}
    cards = []
    for sec, title in PARAM_SECTIONS.items():
        rows = "".join(
            f"<tr><td><code>{p.name}</code></td><td>{esc(str(values[p.name] if values[p.name] is not None else ''))}</td>"
            f"<td>{esc(p.unit)}</td><td><span class='tag {tags[cfg.source[p.name]]}'>{cfg.source[p.name]}</span></td>"
            f"<td>{esc(p.help)}</td></tr>" for p in PARAMS if p.section == sec)
        cards.append(f'<details class="card"><summary>{esc(title)}</summary><table><thead><tr><th>Parameter</th>'
                     f'<th>Value</th><th>Unit</th><th>Source</th><th>Description</th></tr></thead><tbody>{rows}</tbody></table></details>')
    return ('<section id="parameters"><h2>Parameters</h2><p class="intro">Every setting used. <em>Source</em> shows whether a '
            'value is the default or came from the config file (ini) or the command line (cli).</p>' + "".join(cards) + "</section>")


def _command(cfg) -> str:
    cmd = getattr(cfg, "command", "")
    return (f'<section id="command"><h2>Command</h2><p class="intro">The command that produced this report; run it again '
            f'to reproduce the results.</p><div class="card"><pre class="cmd">{esc(cmd)}</pre></div></section>') if cmd else ""


def _glossary() -> str:
    items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in GLOSSARY)
    return f'<section id="glossary"><h2>Glossary</h2><div class="card"><dl>{items}</dl></div></section>'


def build_report(cfg, title="Report") -> str:
    """Assemble the full HTML document."""
    nav = [("overview", "Overview")] + [(s, t) for s, t in SECTIONS.items() if REGISTRY.of(s)] \
        + [("parameters", "Parameters"), ("command", "Command"), ("glossary", "Glossary")]
    body = [_overview()]
    for sec, name in SECTIONS.items():
        items = REGISTRY.of(sec)
        if items:
            cards = "".join(_figure(i) if isinstance(i, Fig) else _block(i) if isinstance(i, Block) else _table(i) for i in items)
            body.append(f'<section id="{sec}"><h2>{name}</h2><p class="intro">{SECTION_INTRO[sec]}</p>{cards}</section>')
    body += [_parameters(cfg), _command(cfg), _glossary()]
    links = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in nav)
    chips = "".join(f'<span class="chip"><small>{esc(k)}</small>{esc(v)}</span>' for k, v in REGISTRY.facts[:4])
    page = (HTML_DIR / "template.html").read_text(encoding="utf-8")
    fill = dict(title=esc(title), css=(HTML_DIR / "style.css").read_text(encoding="utf-8"), nav=links, chips=chips,
                stamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), body="".join(body))
    for key, val in fill.items():
        page = page.replace("{{%s}}" % key, val)
    return page
