"""Stage (optional, off by default): linking of loops in the 3D structures."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.utils import emit_table
from knots.figures import links_approach_figure, links_map_figure, links_stills_figure, links_time_figure
from knots.linking import LinkResult
from logger import get_logger

NAME, ATTR, CACHE = "knots", "links", "links.npz"
log = get_logger(__name__)


def enabled(cfg) -> bool:
    return bool(cfg.detect_knots and cfg.run_md)


def compute(ctx):
    from knots.linking import link_ensemble
    return link_ensemble(ctx.md, ctx.mc_ens, ctx.cfg.knots_stride)


def save(obj, path):
    obj.save(path)


def load(path):
    return LinkResult.load(path)


def describe(ctx):
    r = ctx.links
    c = r.linked_per_frame.astype(float)
    lk = np.abs(r.pairs[:, 4]) if len(r.pairs) else np.zeros(0)
    emit_table("Linking diagnostics", ["Quantity", "Value", "Unit"], [
        ["Frames analysed", len(r.frames), "frames"],
        ["Loops per frame", float(r.n_loops.mean()), "loops"],
        ["Disjoint loop pairs tested", float(r.n_pairs.mean()), "pairs / frame"],
        ["Linked pairs", float(c.mean()), "pairs / frame (mean)"],
        ["Linked pairs, maximum", int(c.max()), "pairs / frame"],
        ["Frames with a link", float((c > 0).mean()), "fraction"],
        ["Linked share of tested pairs", float(c.sum() / max(r.n_pairs.sum(), 1)), "fraction"],
        ["Closest loop approach, median", float(np.median(r.near[:, 2]) / r.bond) if len(r.near) else 0.0, "bond lengths"],
        ["Mean |Lk| of linked pairs", float(lk.mean()) if len(lk) else 0.0, "linking number"],
        ["Pairs with |Lk| >= 2", float((lk >= 2).mean()) if len(lk) else 0.0, "fraction of linked"],
    ], "Pairs of disjoint loops; each loop is closed by the chord between its anchors, linked means |Lk| >= 1.", "knots")
    links_time_figure(r)
    if len(r.near) > 5:
        links_approach_figure(r)
    if len(r.pairs):
        links_map_figure(r)
        if ctx.cfg.render_3d:
            from knots.linking import link_stills
            stills = link_stills(ctx.md.coords, r)
            if stills:
                links_stills_figure(stills)


if __name__ == "__main__":
    from pipeline.runner import run_standalone
    run_standalone(NAME)
