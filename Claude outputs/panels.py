"""Reusable panel drawings (heatmaps, genomic axes)."""
import numpy as np
from matplotlib.colors import LogNorm

from viz.theme import DIV, INK, SEQ


def genomic_x(ax, geom, label=True):
    ax.set_xlim(geom.start / 1e6, geom.end / 1e6)
    if label:
        ax.set_xlabel(f"Position on {geom.chrom} (Mb)")


def heatmap(fig, ax, mat, geom, title, cbar_label, log=False, diverging=False, vmax=None):
    """Contact-style matrix on genomic axes (Mb) with a colour bar."""
    mat = np.asarray(mat, float)
    kw = dict(cmap=DIV if diverging else SEQ, extent=geom.extent_mb(), aspect="equal", interpolation="nearest")
    if log and mat.max() > 0:
        pos = mat[mat > 0]
        im = ax.imshow(mat, norm=LogNorm(vmin=pos.min() * 0.5 if pos.size else 1e-6, vmax=mat.max()), **kw)
    elif diverging:
        v = vmax or np.abs(mat).max() or 1
        im = ax.imshow(mat, vmin=-v, vmax=v, **kw)
    else:
        im = ax.imshow(mat, vmin=0, vmax=vmax or (mat.max() or 1), **kw)
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel(f"Position on {geom.chrom} (Mb)")
    ax.set_ylabel(f"Position on {geom.chrom} (Mb)")
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label(cbar_label, color=INK)
    cb.outline.set_visible(False)
    return im
