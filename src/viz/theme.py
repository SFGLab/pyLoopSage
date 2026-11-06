"""Plot theme, figure factory and saving (white background, purple accents).

- Reusable panel drawings (heatmaps, genomic axes).
"""
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm

from common.utils import moving_average
from reporting.explanations import EXPLAIN
from reporting.report import Fig, REGISTRY


# ---- Plot theme, figure factory and saving (white background, purple accents) ----

matplotlib.use("Agg")


INK, MUTED, GRID = "#111111", "#5a5a66", "#ececf0"
VIOLET, ORANGE, AQUA, MAGENTA = "#6a3de8", "#1a1a1f", "#a98df0", "#4b2bb8"   # names kept: violet, black, light violet, deep violet
EXP_COLOR, SIM_COLOR = ORANGE, VIOLET          # experiment vs simulation, everywhere
MAX_PANELS = 3

SEQ = LinearSegmentedColormap.from_list("loopsage_seq", ["#ffffff", "#ece6fb", "#b9a3f2", "#7a4fe6", "#3d1fa8", "#0d0626"])
# multi-hue heat map colours (white > lilac > violet > magenta > orange > gold): many distinguishable levels
HEAT = LinearSegmentedColormap.from_list("loopsage_heat", ["#ffffff", "#e6dcff", "#a98df0", "#6a3de8", "#9b2fb0", "#e0457b", "#f58a3c", "#ffd45e"])
FREE = LinearSegmentedColormap.from_list("loopsage_free", ["#0d0626", "#3d1fa8", "#6a3de8", "#a98df0", "#e0457b", "#f58a3c", "#ffd45e"])   # no white: every level visible on a white page
COOL = LinearSegmentedColormap.from_list("loopsage_cool", ["#ffffff", "#d9f5ee", "#7fdcc0", "#1baf7a", "#2b62d9", "#5b2fc4", "#0d0626"])   # second multi-hue map
DIV = LinearSegmentedColormap.from_list("loopsage_div", [ORANGE, "#f4f2f8", VIOLET])
# compartments / epigenetic states: B (negative, closed) blue-purple, A (positive, open) red
A_RED, B_BLUE = "#d1303f", "#4a4fd6"
AB = LinearSegmentedColormap.from_list("loopsage_ab", [B_BLUE, "#a9a6ee", "#f1eef6", "#f19aa5", A_RED])


_FONTS = ["Nunito", "Quicksand", "Varela Round"]   # rounded fonts, used when installed


def apply_theme() -> None:
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in _FONTS if f in have] + ["DejaVu Sans"]
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#cfcfd6", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlepad": 10,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": (0, (2, 3)),
        "xtick.major.size": 0, "ytick.major.size": 0, "axes.axisbelow": True, "axes.linewidth": 1.1,
        "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 100,
        "font.family": family,
    })


def glow(ax, x, y, color, lw=1.6, target=300, halo=True, **kw):
    """Smooth line with a soft halo; long series are moving-averaged (raw data kept as a faint trace)."""
    y = np.asarray(y, float)
    window = len(y) // target
    if window > 1:
        ax.plot(x, y, color=color, lw=0.6, alpha=0.18)
        y = moving_average(y, window)
    if halo:
        ax.plot(x, y, color=color, lw=lw * 4, alpha=0.12, solid_capstyle="round")
    return ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", **kw)


def _tag_panels(fig) -> None:
    """Letter the panels A, B, C when a figure has several."""
    axes = [a for a in fig.axes if getattr(a, "_colorbar", None) is None and a.get_label() != "<colorbar>" and not getattr(a, "_twin", False)]
    if len(axes) < 2:
        return
    axes.sort(key=lambda a: (round(a.get_position().x0, 2), -round(a.get_position().y1, 2)))
    for letter, ax in zip("ABC", axes):
        getattr(ax, "text2D", ax.text)(-0.02, getattr(ax, "_tag_y", 1.13), letter, transform=ax.transAxes, fontsize=10, fontweight="bold", color="white", ha="right",
                va="center", bbox=dict(boxstyle="round,pad=0.35", fc=VIOLET, ec="none"))


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


def _display_points(ax):
    """Display-space points of everything drawn in `ax` (lines, markers, filled areas, bars), densified along segments."""
    pts = []
    for ln in ax.lines:
        if ln.get_visible() and len(ln.get_xdata()):
            pts.append(ln.get_transform().transform_path(ln.get_path().interpolated(8)).vertices)
    for c in ax.collections:
        if not c.get_visible():
            continue
        off = c.get_offsets()
        if len(off) and c.get_offset_transform() is not None:
            pts.append(c.get_offset_transform().transform(np.asarray(off, float)))
        for path in c.get_paths():
            if len(path.vertices) > 1:
                pts.append(c.get_transform().transform_path(path).vertices)
    for p in ax.patches:
        if p.get_visible():
            pts.append(p.get_window_extent().corners())
    pts = [q[np.isfinite(q).all(axis=1)] for q in pts if len(q)]
    return np.vstack(pts) if pts else np.empty((0, 2))


def _covers_data(ax, leg) -> bool:
    """Does the legend box sit on top of plotted data?"""
    box, pts = leg.get_window_extent(), _display_points(ax)
    if not len(pts):
        return False
    return bool(((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1) & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)).any())


def _tidy_legends(fig) -> None:
    """Keep every legend clean and off the data: first add head-room on the axis side the legend sits on, else move it under the panel."""
    fig.canvas.draw()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible() or getattr(ax, "name", "") == "3d":
            continue
        leg.set_frame_on(False)
        top, bottom = leg._loc in (1, 2, 9), leg._loc in (3, 4, 8)
        for _ in range(4):
            if not _covers_data(ax, leg):
                break
            if (top or bottom) and ax.get_yscale() == "linear":      # grow the free space around the legend
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo - (hi - lo) * 0.18 * bottom, hi + (hi - lo) * 0.18 * top)
                fig.canvas.draw()
                continue
            leg.set_ncols(min(len(leg.get_texts()), 3))                # last resort: under the panel
            leg.set_loc("upper center")
            leg.set_bbox_to_anchor((0.5, -0.2), transform=ax.transAxes)
            break


def save_figure(fig, key, title=None, caption=None, explain=None, section=None) -> None:
    """Write PNG + SVG and register the figure (with its captions) for the report."""
    if not REGISTRY.draw:                                       # --no-figures: nothing is drawn
        plt.close(fig)
        return
    text = EXPLAIN.get(key, {})
    name = key
    _tag_panels(fig)
    _tidy_legends(fig)
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=170, bbox_inches="tight")
    if REGISTRY.fig_dir is not None:
        (REGISTRY.fig_dir / f"{name}.png").write_bytes(png.getvalue())
        fig.savefig(REGISTRY.fig_dir / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    REGISTRY.add_fig(Fig(name, title or text.get("title", key), caption or text.get("caption", ""),
                         explain or text.get("explain", ""), section or text.get("section", "overview"),
                         png.getvalue()))


# ---- Reusable panel drawings (heatmaps, genomic axes) ----------------------

def genomic_x(ax, geom, label=True):
    ax.set_xlim(geom.start / 1e6, geom.end / 1e6)
    if label:
        ax.set_xlabel(f"Position on {geom.chrom} (Mb)")


def heatmap(fig, ax, mat, geom, title, cbar_label, log=False, diverging=False, vmax=None, cbar=True, gamma=None, cmap=None):
    """Contact-style matrix on genomic axes (Mb) with a colour bar; multi-hue map, robust upper limit."""
    mat = np.asarray(mat, float)
    kw = dict(cmap=cmap or (DIV if diverging else HEAT), extent=geom.extent_mb(), aspect="equal", interpolation="nearest")
    if vmax is None and not diverging and not log:
        vmax = float(np.percentile(mat[mat > 0], 99.5)) if (mat > 0).any() else None       # robust: outliers do not wash out the pattern
    if log and mat.max() > 0:
        pos = mat[mat > 0]
        im = ax.imshow(mat, norm=LogNorm(vmin=pos.min() * 0.5 if pos.size else 1e-6, vmax=mat.max()), **kw)
    elif gamma:
        im = ax.imshow(mat, norm=PowerNorm(gamma, vmin=0, vmax=vmax or (mat.max() or 1)), **kw)
    elif diverging:
        v = vmax or np.abs(mat).max() or 1
        im = ax.imshow(mat, vmin=-v, vmax=v, **kw)
    else:
        im = ax.imshow(mat, vmin=0, vmax=vmax or (mat.max() or 1), **kw)
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel(f"Position on {geom.chrom} (Mb)")
    ax.set_ylabel(f"Position on {geom.chrom} (Mb)")
    if not cbar:
        return im
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(cbar_label, color=INK)
    cb.outline.set_visible(False)
    return im
