"""Collects figures, tables and key facts as stages run; rendered at the end."""
from dataclasses import dataclass, field
from pathlib import Path

SECTIONS = {
    "input": "Input data",
    "monte_carlo": "Loop extrusion (Monte Carlo)",
    "molecular_dynamics": "3D structures (molecular dynamics)",
    "validation": "Validation against experiment",
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
class Registry:
    items: list = field(default_factory=list)      # Fig | Tab, in creation order
    facts: list = field(default_factory=list)      # (label, value) headline numbers
    timings: list = field(default_factory=list)    # (stage, seconds)
    fig_dir: Path = None

    def reset(self, fig_dir=None):
        self.items, self.facts, self.timings, self.fig_dir = [], [], [], fig_dir

    def add_table(self, title, columns, rows, caption, section):
        self.items.append(Tab(title, list(columns), rows, caption, section))

    def add_fig(self, fig: Fig):
        self.items.append(fig)

    def fact(self, label, value):
        self.facts.append((label, str(value)))

    def of(self, section):
        return [i for i in self.items if i.section == section]


REGISTRY = Registry()
