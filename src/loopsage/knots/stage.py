"""Stage (optional, off by default): linking of loops in the 3D structures."""

import numpy as np

from loopsage.common.utils import emit_table
from loopsage.knots.crossing import barrier_kT, strength_for_barrier
from loopsage.knots.figures import crossings_figure, links_approach_figure, links_dynamics_figure, links_examples_figure, links_map_figure, links_stills_figure, links_time_figure
from loopsage.knots.linking import LinkResult
from loopsage.logger import get_logger
from loopsage.reporting.report import REGISTRY

NAME, ATTR, CACHE = "knots", "links", "links.npz"
log = get_logger(__name__)


def enabled(cfg) -> bool:
    return bool((cfg.detect_knots or cfg.track_crossings or cfg.topo_mode == "local") and cfg.run_md)


def compute(ctx):
    """Linking numbers (only with detect_knots); otherwise an empty result, the crossing counter lives in the MD."""
    from loopsage.knots.linking import link_ensemble
    if not ctx.cfg.detect_knots:
        return LinkResult.empty(ctx.md.coords.shape[1])
    return link_ensemble(ctx.md, ctx.mc_ens, ctx.cfg.knots_stride)


def save(obj, path):
    obj.save(path)


def load(path):
    return LinkResult.load(path)


def _crossings(ctx):
    """Excluded-volume barrier and the chain crossings counted during the MD (tables and one figure)."""
    cfg, topo = ctx.cfg, ctx.md.meta.get("topology")
    bar = barrier_kT(cfg.ev_ff_strength, cfg.ev_ff_power, cfg.sim_temp)
    sug = strength_for_barrier(cfg.ev_barrier_target, cfg.ev_ff_strength, cfg.ev_ff_power, cfg.sim_temp)
    emit_table("Strand-crossing barrier of the excluded volume", ["Quantity", "Value", "Unit"], [
        ["EV strength / exponent", f"{cfg.ev_ff_strength:g} / {cfg.ev_ff_power:g}", "kJ/mol / -"],
        ["Barrier for two bonds to pass through each other", bar, "kT"],
        [f"ev_ff_strength for a {cfg.ev_barrier_target:g} kT barrier", sug, "kJ/mol"],
        ["Thermal energy kT", 0.008314 * cfg.sim_temp, "kJ/mol"]],
        "Lowest energy of the four bead pairs when two bonds intersect (a lower bound). A few kT is crossed easily; about 10 kT or more keeps the chain intact. "
        "The suggested strength is for studies of topology; the default model is unchanged.", "knots")
    REGISTRY.fact("EV crossing barrier", f"{bar:.1f} kT")
    if not topo:
        return
    steps = cfg.sim_step * len(topo["leak"])
    rows = [["Mode", topo["mode"] if topo["mode"] == "local" else f"random (ev_p = {topo['ev_p']:g})", ""],
            ["Crossings with the EV on (leaks)", int(sum(topo["leak"])), f"{1000 * sum(topo['leak']) / max(steps, 1):.3g} per 1000 MD steps"],
            ["Crossings through a switched-off bead (passages)", int(sum(topo["passage"])), ""]]
    if topo["mode"] == "local":
        rows += [["Passage attempts", int(sum(topo["events"])), "events"], ["Undone by the topology-simplifying rule", int(sum(topo["rejected"])), f"topo_keep = {topo['keep']:g}"],
                 ["Links created / removed by the attempts", f"{int(sum(topo['created']))} / {int(sum(topo['removed']))}", "loop pairs"]]
    emit_table("Chain crossings during the MD", ["Quantity", "Value", "Note"], rows,
               f"Two bonds passing through each other between checks every {topo['track_every']} MD steps. Leaks show that the excluded volume lets strands cross even without the topoisomerase.", "knots")
    curve_x = np.geomspace(cfg.ev_ff_strength / 4, cfg.ev_ff_strength * 4, 25)
    crossings_figure(topo, bar, curve_x, [barrier_kT(x, cfg.ev_ff_power, cfg.sim_temp) for x in curve_x], cfg.ev_barrier_target, cfg.ev_ff_strength)


def describe(ctx):
    r = ctx.links
    _crossings(ctx)
    if not len(r.frames):
        return
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
    links_dynamics_figure(r)
    if len(r.near) > 5:
        links_approach_figure(r)
    if len(r.pairs):
        links_map_figure(r)
        if ctx.cfg.render_3d:
            from loopsage.knots.linking import link_examples, link_stills
            stills = link_stills(ctx.md.coords, r)
            if stills:
                links_stills_figure(stills)
            ex = link_examples(ctx.md.coords, r)
            if ex:
                links_examples_figure(ex)


if __name__ == "__main__":
    from loopsage.pipeline.runner import run_standalone
    run_standalone(NAME)
