"""Plot theme, figure factory and saving (white background, purple accents).

Categorical colours: violet, orange, aqua (validated together for colour-blind
safety). Rule enforced here: no figure has more than MAX_PANELS panels.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from reporting.explanations import EXPLAIN
from reporting.registry import REGISTRY, Fig

INK, MUTED, GRID = "#2b2540", "#6b6485", "#ece8f5"
VIOLET, ORANGE, AQUA, MAGENTA = "#6250d6", "#eb6834", "#1baf7a", "#e87ba4"
EXP_COLOR, SIM_COLOR = ORANGE, VIOLET          # experiment vs simulation, everywhere
MAX_PANELS = 3

SEQ = LinearSegmentedColormap.from_list("loopsage_seq", ["#ffffff", "#ebe3fb", "#b9a1f0", "#7a5ce0", "#3a1f85"])
DIV = LinearSegmentedColormap.from_list("loopsage_div", [ORANGE, "#f4f2ee", VIOLET])


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 100,
        "font.family": ["Nunito", "DejaVu Sans"],
    })


def make_fig(nrows: int = 1, ncols: int = 1, cell=(5.4, 4.2), **kw):
    """Grid of panels; returns (fig, flat list of axes)."""
    assert nrows * ncols <= MAX_PANELS, f"at most {MAX_PANELS} panels per figure"
    apply_theme()
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows),
                             squeeze=False, layout="constrained", **kw)
    return fig, list(axes.ravel())


def make_mosaic(layout, size=(10.5, 4.6), **kw):
    """Custom panel layout (subplot_mosaic); returns (fig, {label: axes})."""
    labels = {c for row in layout for c in row}
    assert len(labels) <= MAX_PANELS, f"at most {MAX_PANELS} panels per figure"
    apply_theme()
    fig = plt.figure(figsize=size, layout="constrained")
    return fig, fig.subplot_mosaic(layout, **kw)


def save_figure(fig, key, title=None, caption=None, explain=None, section=None) -> None:
    """Write PNG + SVG and register the figure (with its captions) for the report."""
    text = EXPLAIN.get(key, {})
    name = key
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=170, bbox_inches="tight")
    if REGISTRY.fig_dir is not None:
        (REGISTRY.fig_dir / f"{name}.png").write_bytes(png.getvalue())
        fig.savefig(REGISTRY.fig_dir / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    REGISTRY.add_fig(Fig(name, title or text.get("title", key), caption or text.get("caption", ""),
                         explain or text.get("explain", ""), section or text.get("section", "overview"),
                         png.getvalue()))
