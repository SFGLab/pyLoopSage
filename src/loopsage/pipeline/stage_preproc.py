"""Stage 1: read experimental input and describe it."""
import sys
from pathlib import Path

import numpy as np

def _motif(ctx):
    """CTCF motif search: parameters, loop/anchor counts and two diagnostic figures. Returns the loops file to analyse."""
    from loopsage.motif_finder import MotifResult, random_hit_rate
    from loopsage.viz.motif_figs import search_figure
    res = ctx.motif or MotifResult.load(ctx.dirs.data)
    if res is None:
        return ctx.loops_file
    st = res.stats
    rows = [["Minimum PET count", st["min_pet"], "PET" if st["has_pet"] else "no PET column: filter skipped"],
            ["Loops in file", st["n_file"], f"{st['malformed']} unreadable lines skipped" if st["malformed"] else ""],
            ["Loops inside the region", st["n_region"], ""], ["Loops kept", st["n_kept"], "unique loops"]]
    if st["scanned"]:
        a = res.anchors
        rnd = random_hit_rate(st, a)
        got = st["n_with_motif"] / max(st["n_scanned"], 1)
        rows += [["Motif", f"{st['motif']} ({st['consensus']})", f"{st['motif_length']} bp"],
                 ["Score cutoff", st["cutoff"], f"p <= {st['pvalue']:g}; best possible {st['max_score']:.1f}"],
                 ["Distinct anchors scanned", st["n_scanned"], f"{st['n_anchors'] - st['n_scanned']} unreadable"],
                 ["Anchors with a motif", f"{got:.0%}", f"random DNA: {rnd:.0%} (enrichment x{got / max(rnd, 1e-9):.1f})"],
                 ["Loops with both anchors oriented", st["n_both"], ""], ["Loops with one anchor oriented", st["n_one"], "the other is split evenly"],
                 ["Loops with no oriented anchor", st["n_none"], "orientation unknown"]]
        REGISTRY.fact("CTCF motif search", f"{got:.0%} of anchors with a motif")
    emit_table("CTCF motif search" if st["scanned"] else "Loop filtering", ["Quantity", "Value", "Note"], rows,
               "Loops are filtered by PET count, then each distinct anchor is scanned for the CTCF motif on both strands; "
               "the orientation probability is written to the annotated BEDPE (data folder).", "input")
    if st["scanned"]:
        search_figure(res)
    return str(res.path)


def _anchors(ctx, path):
    """Tables and figures about anchor strength, CTCF orientation and its confidence."""
    a = anchor_table(path, ctx.geom)
    if a is None:
        return
    emit_table("Anchor orientation", ["Orientation", "Anchors", "Share", "Median strength", "Mean confidence"],
               [[r[0], r[1], f"{r[2]:.0%}", r[3], r[4]] for r in orientation_summary(a)],
               "Forward = P(reverse) < 0.5; ambiguous = confidence |2p-1| below 0.2; unknown = no call in the file.", "input")
    emit_table("Loop configurations", ["Configuration", "Loops", "Share", "Mean length (kb)", "Mean strength"],
               [[r[0], r[1], f"{r[2]:.0%}", r[3], r[4]] for r in config_summary(a)],
               "Convergent = left anchor forward and right anchor reverse (the loop-extrusion rule).", "input")
    conv = float((a.config == "convergent").mean())
    REGISTRY.fact("Convergent loops", f"{conv:.0%}")
    input_figs.anchor_map(a, ctx.geom)
    input_figs.anchor_stats(a)
    input_figs.anchor_loops(a)


def _tracks(ctx):
    """Summary table of the extra tracks (BigWig, LEF loading)."""
    inp = ctx.inputs
    tracks = list(zip(inp.bw_names, inp.BWs)) + ([("LEF loading", inp.lef_track)] if inp.lef_track is not None else [])
    if not tracks:
        return
    emit_table("Input tracks", ["Track", "Mean", "Std", "Min", "Max", "Spearman rho with anchors"],
               [[n, float(v.mean()), float(v.std()), float(v.min()), float(v.max()), track_rho(v, inp)] for n, v in tracks],
               "Signals averaged per bead (a.u.); rho compares each track with the smoothed input anchor strength.", "input")


def _compartments(ctx):
    """Compartment summary table and figure."""
    sig = ctx.inputs.loops.L + ctx.inputs.loops.R
    summ = compartment_summary(ctx.inputs.comp, sig, ctx.geom)
    emit_table("Compartments", ["Quantity", "Value", "Unit"], [
        ["Compartment A", f"{summ['frac_a']:.0%}", "of beads"], ["Compartment B", f"{1 - summ['frac_a']:.0%}", "of beads"],
        ["Domains", summ["n_domains"], "count"],
        ["Median A domain", float(np.median(summ["size_a"])) if summ["size_a"].size else 0.0, "kb"],
        ["Median B domain", float(np.median(summ["size_b"])) if summ["size_b"].size else 0.0, "kb"],
        ["Mean anchor strength in A", float(summ["anchors_a"].mean()) if summ["anchors_a"].size else 0.0, "a.u. per bead"],
        ["Mean anchor strength in B", float(summ["anchors_b"].mean()) if summ["anchors_b"].size else 0.0, "a.u. per bead"],
        ["Anchors A vs B (Mann-Whitney p)", summ["p"], ""],
    ], "A = positive compartment signal, B = negative; short flickers (1 bead) are merged into their neighbours.", "input")
    REGISTRY.fact("Compartment A", f"{summ['frac_a']:.0%}")
    compartment_figure(ctx.inputs.comp, sig, summ, ctx.geom)



from loopsage.analysis.inputs import anchor_table, config_summary, orientation_summary
from loopsage.common.genome import fmt_bp
from loopsage.common.runtime import stage_bar
from loopsage.common.utils import describe_row, emit_table
from loopsage.analysis.maps import compartment_from_hic
from loopsage.logger import get_logger
from loopsage.preproc.loader import load_inputs
from loopsage.preproc.loader import InputData
from loopsage.reporting.report import REGISTRY
from loopsage.analysis.inputs import compartment_summary
from loopsage.viz import input_figs
from loopsage.viz.input_figs import compartment_figure
from loopsage.viz.input_figs import binding_profiles, loop_matrices, track_figures, track_rho

log = get_logger(__name__)

NAME, ATTR, CACHE = "preproc", "inputs", "inputs.npz"


def enabled(cfg) -> bool:
    return True


def _input_use(ctx):
    """Table saying which input maps (L/R, J, J_loss) the simulation actually uses, and how they were processed."""
    cfg, g = ctx.cfg, ctx.geom
    two_d = int(cfg.data_loss_mode) == 1
    epi = float(cfg.epi_coeff) > 0
    sigma = max(1.5, g.n_beads / 300)
    smooth = "Gaussian smoothing (sigma = %.1f beads)" % sigma if cfg.smoothing_input else "no smoothing"
    proc = smooth
    proc_loss = "smoothed to a continuous landscape (sigma = %.1f beads)" % float(cfg.loss_sigma) if cfg.loss_sigma > 0 else "not smoothed"
    dyn = str(cfg.epigenetic_mode).lower() == "dynamic"
    rows = [["Anchor profiles L, R", "USED: data energy at the left / right legs" if not two_d else "not used (computed for display)", proc],
            ["Loop map J_loss", "USED: data energy of every loop (m, n)" if two_d else "not used (computed for display)", proc_loss if two_d else ""],
            *([["Loop coverage", "USED (contrastive): loops are rewarded for spanning covered beads, penalised on loop-free ones", "-1 on loop-free beads (coverage < 5% of the mean), else (loops spanning the bead / mean - 1) clipped to [0, 1]"]] if cfg.binding_mode == "contrastive" else []),
            ["Loop adjacency J", ("USED: epigenetic coupling" + (" (replaced by the current LEF loops in 'dynamic' mode)" if dyn else "")) if epi else "not used (epi_coeff = 0)", "binary, never smoothed"]]
    emit_table("What the simulation uses as input", ["Input", "Role in the simulation", "Processing"], rows,
               "data_loss_mode = %d selects %s. J_loss is the loop strength map; in the 2D mode it is a smooth landscape that the LEF legs move on." % (int(cfg.data_loss_mode), "the 2D loop map J_loss" if two_d else "the 1D profiles L and R"), "input")
    REGISTRY.fact("Data energy uses", "J_loss (2D loop map)" if two_d else "L and R (1D profiles)")


def _hic(ctx):
    """Optional Hi-C reference: what was read, and the map with its decay and insulation."""
    from loopsage.viz.hic_figs import hic_input_figure
    H, info = ctx.hic
    emit_table("Hi-C reference", ["Quantity", "Value", "Unit"],
               [["File", info["file"], ".hic"], ["Resolution read", info["resolution"], "bp"], ["Normalisation", info["norm"], ""],
                ["Beads covered", info["coverage"], "fraction"], ["Used for", "validation" + (" and the optimizer target" if cfg_opt(ctx) else ""), ""]],
               "The Hi-C map of the region read with hic-straw and put on the bead grid of the model (masked beads stay empty).", "input")
    REGISTRY.fact("Hi-C", f"{info['resolution'] // 1000} kb, {info['norm']}")
    hic_input_figure(H, ctx.geom, info, ctx.cfg.track_smooth)


def cfg_opt(ctx) -> bool:
    return bool(ctx.cfg.optimize)


def compute(ctx):
    """Optional BEDPE preparation (PET filter, CTCF motif search), then read the loops."""
    cfg, g = ctx.cfg, ctx.geom
    path = cfg.interaction_file
    if cfg.motif_genome or cfg.min_pet:
        from loopsage.motif_finder import annotate_bedpe
        res = annotate_bedpe(path, ctx.dirs.data / "loops_annotated.bedpe", (g.chrom, g.start, g.end), cfg.motif_genome, cfg.motif_pfm,
                             cfg.motif_pvalue, cfg.motif_pseudocount, cfg.min_pet, stage_bar, cfg.motif_pad)
        res.save(ctx.dirs.data)
        ctx.motif, path = res, str(res.path)
    ctx.input_file = path
    inputs = load_inputs(cfg, g, path)
    if inputs.comp is None and float(cfg.epi_coeff) > 0 and ctx.hic is not None:      # Potts field h from the Hi-C compartments (B = negative)
        inputs.comp, inputs.comp_source = compartment_from_hic(ctx.hic[0], smooth=cfg.track_smooth), "hic"
        log.info("No compartment track given: the Potts field h is the leading eigenvector of the Hi-C O/E correlation (B = denser = negative).")
    return inputs


def load(path):
    return InputData.load(path)


def describe(ctx):
    g, lo, cfg = ctx.geom, ctx.inputs.loops, ctx.cfg
    REGISTRY.fact("Region", g.label)
    REGISTRY.fact("Resolution", f"{fmt_bp(g.bp_per_bead)} / bead")
    emit_table("Input summary", ["Quantity", "Value", "Unit"], [
        ["Region", g.label, ""], ["Region length", fmt_bp(g.length_bp), "bp"],
        ["Simulation beads", g.n_beads, "beads"], ["Resolution", fmt_bp(g.bp_per_bead), "bp per bead"],
        ["Input file", Path(cfg.interaction_file).name, lo.fmt],
        ["Anchors / sites", int(lo.n_sites), "count"],
        *[[k.replace("_", " ").capitalize(), v, ""] for k, v in lo.stats.items() if isinstance(v, (int, float))],
        ["Extra tracks", len(ctx.inputs.bw_names), "BigWig"],
    ], "What was read from the input file and how it maps to simulation beads.", "input")
    if len(lo.loop_lengths):
        kb = g.bp_per_bead / 1e3
        emit_table("Input loop lengths", ["Quantity", "Mean", "Median", "Min", "Max", "Unit"],
                   [describe_row("Loop length", lo.loop_lengths, "kb", kb)],
                   "Distance between the two anchors of each unique input loop.", "input")
        REGISTRY.fact("Input loops", len(lo.loop_lengths))
    _input_use(ctx)
    if ctx.hic is not None:
        _hic(ctx)
    if lo.fmt == "bedpe":
        _anchors(ctx, _motif(ctx))
    binding_profiles(ctx.inputs, g)
    loop_matrices(ctx.inputs, g)
    _tracks(ctx)
    track_figures(ctx.inputs, g)
    if ctx.inputs.comp is not None:
        _compartments(ctx)


if __name__ == "__main__":
    from loopsage.pipeline.runner import run_standalone
    run_standalone(NAME)
