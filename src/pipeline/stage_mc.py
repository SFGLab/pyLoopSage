"""Stage 2: Monte Carlo loop extrusion and its diagnostics."""
import sys
from pathlib import Path


def _epigenetic(ctx, mc, g):
    """Figures and summary table of the (optional) epigenetic model."""
    from analysis import ensemble
    from monte_carlo.sampler import build_params
    cfg, n = ctx.cfg, ctx.cfg.n_epi_states
    p = build_params(cfg, ctx.inputs, g.n_beads, mc.meta["n_lef"])
    steps, energy = ensemble.epigenetic_energy_series(p, mc)
    mc_figs.epi_trajectory(mc, g)
    _epigenetic_maps(ctx, mc, g)
    mc_figs.epigenetic_model(mc, g, steps, energy, ctx.inputs.comp, n)
    mc_figs.epigenetic_loops(mc, g, n)
    ag = ensemble.loop_state_agreement(mc, n)
    emit_table("Epigenetic model", ["Quantity", "Value", "Unit"], [
        ["Coupling mode", cfg.epigenetic_mode, ""], ["Coupling strength", cfg.epi_coeff, "a.u."], ["Epigenetic states", n, ""],
        ["Loop ends sharing a state", ag["overall"], "fraction"], ["Same for any bead pair at that distance", ag["overall_base"], "fraction"],
    ], "Loops that join beads of the same epigenetic state raise the first fraction above the second.", "monte_carlo")


def _epigenetic_maps(ctx, mc, g):
    """Loop map x epigenetic checkerboard; the combined map replaces the loop-only map for validation."""
    from analysis.maps import checkerboard, combine_maps, epigenetic_profile, map_features
    n = ctx.cfg.n_epi_states
    prof = ctx.mc_epi_profile = epigenetic_profile(mc, n)
    board = checkerboard(prof)
    comb = combine_maps(ctx.mc_map, board)
    f_loop, f_comb = ctx.features["mc"], map_features(comb, ctx.inputs.comp)
    mc_figs.epigenetic_maps(ctx.mc_map, board, comb, g)
    mc_figs.epigenetic_features(prof, g, f_loop, f_comb, ctx.inputs.comp)
    ctx.mc_loop_map, ctx.mc_map = ctx.mc_map, comb
    ctx.features["mc_epi"] = f_comb


def _speed(ctx, mc, g):
    """LEF velocities as statistical physics: table and three figures."""
    from analysis.speed import lef_windows, msd, speed_summary, velocities
    from viz.speed_figs import free_energy_figure, phase_space_figure, speed_init_figure, surface_figure, velocity_figure
    speed_init_figure(mc)
    w = lef_windows(mc)
    if len(w) < 10 or min(c.n_samples - c.burnin_idx for c in mc.chains) < 30:
        log.info("Too few snapshots / rebinding-free windows for the speed analysis (%d windows)", len(w))
        return
    v = velocities(mc, ctx.inputs)
    sm, kb = speed_summary(w, v, msd(mc)), g.bp_per_bead / 1e3
    mode = ("random-walk sliding" if mc.meta.get("lef_rw", True) else "outward extrusion") + f", {mc.meta.get('lef_speed_dist', 'constant')} speeds"
    rows = [["Mode", mode, ""], ["Windows analysed", sm["n_windows"], "count"],
            ["Mean speed v (each leg moves at +/- v)", float(v["v"].mean()) * kb, "kb per MC step"],
            ["Root-mean-square speed", np.sqrt(sm["temp"]) * kb, "kb per MC step"], ["Legs at rest between snapshots", sm["rest"], "fraction"],
            ["Excess kurtosis of the signed velocity (0 = Gaussian)", sm["kurt"], ""], ["MSD exponent of loop length", sm["alpha"], "1 = diffusion, 2 = ballistic"],
            ["Spread of speeds between LEFs (CV)", sm["lef_cv"], "std / mean"], ["Loop length vs speed", sm["rho_length"], "Spearman rho"]]
    if not np.isnan(sm["exponent"]):
        rows.append(["Measured vs set speed exponent", sm["exponent"], "0.5 diffusion, 1 extrusion"])
    emit_table("LEF velocities", ["Quantity", "Value", "Unit"], rows,
               "Velocities of the loop length between snapshots (rebinding jumps excluded); windows of 8 snapshots for phase space.", "monte_carlo")
    velocity_figure(mc, v, sm, g)
    phase_space_figure(w, v, mc, ctx.inputs, g)
    free_energy_figure(w, mc, ctx.inputs, g)
    surface_figure(w, g)


def _nesting(ctx, mc, g):
    """Parents versus nested children: table and two figures."""
    from analysis.nesting import nesting_summary, production_nesting
    from viz.nesting_figs import nesting_figure, parents_children_figure
    parts = production_nesting(mc)
    if not parts or parts[0].valid.shape[1] < 3:
        return
    s, kb = nesting_summary(parts), g.bp_per_bead / 1e3
    rows = [["Fraction of loops that are parents (hold loops, inside none)", s["parent_fraction"], ""],
            ["Fraction of loops that are orphans (isolated)", s["orphan_fraction"], ""], ["Mean nesting order, all loops", s["mean_order"], "order"],
            ["Mean order of children only", s["mean_child_order"], "order"], ["Deepest order seen", s["max_order"], "order"],
            ["Mean deepest order per snapshot", s["mean_max_order"], "order"], ["Parents per snapshot", s["parents_per_snapshot"], "loops"], ["Orphans per snapshot", s["orphans_per_snapshot"], "loops"],
            ["Loops per snapshot", s["loops_per_snapshot"], "loops"], ["Loops inside a parent (all orders)", s["descendants_per_parent"], "loops"],
            ["Child length / smallest enclosing loop length", s["length_ratio"], "ratio"]]
    rows += [[f"Mean length, order {k}" + ("+" if k == 3 else ""), s["mean_length"][k] * kb, "kb"] for k in range(4)]
    emit_table("Loop nesting", ["Quantity", "Value", "Unit"], rows,
               "A parent holds loops and is in none; an orphan is isolated (in no loop, holds none); a loop inside k others has order k. Post-burn-in snapshots of all chains.", "monte_carlo")
    nesting_figure(mc, parts)
    parents_children_figure(mc, parts, g)


def _replicas(mc, g):
    """Table and figure for the agreement of independent chains."""
    from analysis.ensemble import replica_summary
    from viz.mc_figs import replica_figure
    s = replica_summary(mc)
    emit_table("Replicas", ["Replica", "Seed", "Mean energy", "Std", "Unit"],
               [[k, c.meta["seed"], float(s["mean"][k]), float(s["std"][k]), "a.u."] for k, c in enumerate(mc.chains)],
               f"{len(mc.chains)} independent chains; R-hat = {s['rhat']:.3f} (close to 1 = agreement), "
               f"lowest pairwise occupancy correlation = {s['min_corr']:.2f}.", "monte_carlo")
    REGISTRY.fact("Replicas", f"{len(mc.chains)} (R-hat {s['rhat']:.2f})")
    replica_figure(mc, g, s)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from analysis import ensemble
from viz.map_figs import map_feature_report
from analysis.ensemble import acceptance_table, loop_lengths, occupancy
from analysis.ensemble import equilibrium_summary
from analysis.maps import loop_counts, loop_map
from common.genome import fmt_duration
from common.utils import describe_row, emit_table
from logger import get_logger
from monte_carlo.simulation import MCResult
from optimizer.objective import COMMUNICATION
from monte_carlo.simulation import run_mc
from reporting.report import REGISTRY
from viz import mc_figs

log = get_logger(__name__)
NAME, ATTR, CACHE = "mc", "mc", "mc.npz"
_TERM = dict(Es="Total", Bs="Binding", Ks="Crossing", Fs="Folding")


def enabled(cfg) -> bool:
    return True


def compute(ctx):
    if getattr(ctx, "optimization", None) is not None:
        log.step("Running the full Monte Carlo (all %d steps) with the optimized parameters", ctx.cfg.n_steps)
    return run_mc(ctx.cfg, ctx.inputs, ctx.geom)


def load(path):
    return MCResult.load(path)


def describe(ctx):
    mc, g = ctx.mc, ctx.geom
    m, kb = mc.meta, ctx.geom.bp_per_bead / 1e3
    REGISTRY.fact("LEFs", m["n_lef"] + m["n_lef2"])
    REGISTRY.fact("MC steps", f"{m['n_steps']:,}")
    REGISTRY.fact("LEF speeds", f"{m.get('lef_speed_dist', 'constant')}" + ("" if m.get("lef_rw", True) else ", extrusion"))
    emit_table("Monte Carlo run", ["Quantity", "Value", "Unit"], [
        ["Method", m["method"], ""], ["LEFs", m["n_lef"] + m["n_lef2"], "count"],
        ["MC steps", m["n_steps"], "steps"], ["Burn-in", m["burnin"], "steps"],
        ["Snapshots saved", mc.n_samples, f"every {m['mc_step']} steps"],
        ["Snapshots after burn-in", mc.n_samples - mc.burnin_idx, "count"],
        ["Runtime", fmt_duration(m["runtime_s"]), f"{m.get('replicas', 1)} replica(s), {m.get('threads', 1)} thread(s)"],
    ], "Sampler settings; steps are Monte Carlo moves, not physical time.", "monte_carlo")

    eq = equilibrium_summary(mc)
    emit_table("Energy after burn-in", ["Term", "Mean", "Std", "Drift", "Unit"],
               [[_TERM[k], eq[k]["mean"], eq[k]["std"], eq[k]["drift"], "a.u. (drift in std)"] for k in _TERM],
               "Drift = difference of the two halves in units of the standard deviation; |drift| well below 1 means stable.",
               "monte_carlo", level=1)
    emit_table("Convergence", ["Quantity", "Value", "Unit"], [
        ["Autocorrelation time", eq["tau"] * m["mc_step"], "MC steps"],
        ["Independent snapshots (ESS)", eq["ess"], "count"], ["Snapshots used", eq["n"], "count"],
    ], "Autocorrelation time of the folding energy and the resulting effective sample size.", "monte_carlo")
    emit_table("Move acceptance", ["Move", "Proposed", "Accepted", "Rate"],
               [list(r) for r in acceptance_table(mc)], "Fraction of proposed moves that were accepted.", "monte_carlo")
    emit_table("Simulated loops", ["Quantity", "Mean", "Median", "Min", "Max", "Unit"],
               [describe_row("Loop length", loop_lengths(mc), "kb", kb)],
               "Loop lengths pooled over LEFs and post-burn-in snapshots.", "monte_carlo", level=1)

    rec, dyn = ensemble.anchor_recovery(mc, ctx.inputs), ensemble.lef_dynamics(mc)
    emit_table("Anchor recovery", ["Quantity", "Value", "Unit"], [
        ["Strong input anchors", rec["n_anchor_beads"], "beads"], ["Well-occupied beads", rec["n_used_beads"], "beads"],
        ["Anchors held by a LEF (recall)", rec["recall"], "fraction"], ["Occupied beads that are anchors (precision)", rec["precision"], "fraction"],
        ["Occupancy vs anchor strength", rec["spearman"], "Spearman rho"],
    ], "Strong anchor = above the 75th percentile of the non-zero input signal; well-occupied = at least half the maximum occupancy.", "monte_carlo")
    emit_table("LEF dynamics", ["Quantity", "Value", "Unit"], [
        ["Unchanged between snapshots", dyn["stable_fraction"], "fraction"], ["Typical dwell time", dyn["dwell_steps"], "MC steps"],
        ["Rebinding jumps", dyn["rebinding_per_1000"], "per LEF per 1000 steps"], ["LEFs never moving after burn-in", dyn["frozen_lefs"], "fraction"],
    ], "How mobile the loops are: high stability and long dwell times mean loops are locked on anchors.", "monte_carlo")
    REGISTRY.fact("Anchor recall", f"{rec['recall']:.0%}")
    ctx.mc_map = loop_map(ctx.mc_ens, g.n_beads, **(COMMUNICATION if ctx.cfg.optimize else {}))       # all replicas pooled; the map the optimizer scored (same communication)
    if ctx.cfg.save_metadata:
        meta = ctx.dirs.data / "metadata"
        meta.mkdir(exist_ok=True)
        np.save(meta / "mc_loop_counts.npy", loop_counts(ctx.mc_ens, g.n_beads))      # mean loops joining i and j
        np.save(meta / "mc_contact_map.npy", ctx.mc_map)                               # loop map with communication
    mc_figs.energy_traces(mc)
    mc_figs.convergence(mc)
    mc_figs.lef_trajectories(mc, g)
    mc_figs.kymograph(mc, g, ctx.inputs)
    mc_figs.kymograph_zoom(mc, g, ctx.cfg.zoom_snapshots, ctx.cfg.zoom_start)
    mc_figs.occupancy_lengths(mc, g, ctx.inputs)
    mc_figs.loop_landscape(mc, g, ctx.inputs)
    mc_figs.mc_heatmap(ctx.mc_map, g)
    map_feature_report(ctx, "mc", ctx.mc_map, "Loop ensemble (MC)")
    _speed(ctx, mc, g)
    _nesting(ctx, mc, g)
    if mc.replicas:
        _replicas(mc, g)
    if m.get("epi_on"):                                           # epigenetic model: off unless epi_coeff != 0
        _epigenetic(ctx, mc, g)


if __name__ == "__main__":
    from pipeline.runner import run_standalone
    run_standalone(NAME)
