"""Stage 3: 3D polymer molecular dynamics, one independent run per replica."""
from loopsage.analysis.replicas import map_correlation, map_spread, mean_offdiag
from loopsage.common.genome import fmt_duration
from loopsage.common.utils import describe_row, emit_table
from loopsage.molecular_dynamics.engine import MDResult, run_md_replicas
from loopsage.reporting.report import REGISTRY
from loopsage.viz import map_figs, replica_figs
from loopsage.viz.map_figs import map_feature_report

NAME, ATTR, CACHE = "md", "md", "md.npz"


def enabled(cfg) -> bool:
    return bool(cfg.run_md)


def compute(ctx):
    return run_md_replicas(ctx.cfg, ctx.geom, ctx.mc, ctx.dirs)


def load(path):
    return MDResult.load(path)


def _replicas(ctx):
    """Table and figures for the agreement of the independent MD runs (several replicas only)."""
    md, g = ctx.md, ctx.geom
    corr = map_correlation([c.avg_map for c in md.chains])
    emit_table("MD replicas", ["Replica", "Seed", "Frames", "Mean R_g", "Mean potential energy"],
               [[k, c.meta["seed"], len(c.rg), f"{c.rg.mean():.4g} nm", f"{c.epot.mean():.5g} kJ/mol"] for k, c in enumerate(md.chains)],
               f"{len(md.chains)} independent MD runs, each from its own Monte Carlo chain, start structure and random stream; "
               f"mean pairwise correlation of their contact maps = {mean_offdiag(corr):.3f}.", "molecular_dynamics")
    REGISTRY.fact("MD map agreement", f"r = {mean_offdiag(corr):.2f}")
    replica_figs.replica_maps_figure("md_replica_maps", "3D contact map", [c.avg_map for c in md.chains], g, "Contact probability (0-1)", "molecular_dynamics")
    replica_figs.md_distance_figure([c.avg_dist for c in md.chains], g)
    replica_figs.md_traces_figure(md)


def _per_replica(ctx):
    """The MD figures of every replica, saved in its own visualizations/replica_<k> folder (not in the report)."""
    for k, md in enumerate(ctx.md.chains):
        with REGISTRY.scope(ctx.dirs.replica(k).figures):
            map_figs.md_maps(md, ctx.geom)
            map_figs.md_series(md)
            map_figs.md_geometry(md, ctx.mc.chains[k], ctx.geom)
            if ctx.mc_replica_maps:
                map_figs.mc_vs_md(ctx.mc_replica_maps[k], md.avg_map, ctx.geom)


def describe(ctx):
    md, m = ctx.md_ens, ctx.md_ens.meta
    n = m["n_frames"]
    REGISTRY.fact("MD frames", n)
    emit_table("Molecular dynamics run", ["Quantity", "Value", "Unit"], [
        ["Platform", m["platform"], m.get("platform_detail", "")], ["Frames", n, "structures"],
        ["Time step", m["dt_fs"], "fs"], ["Steps per frame", m["steps_per_frame"], "MD steps"],
        ["Time per frame", m["ps_per_frame"], "ps"], ["Total simulated time", n * m["ps_per_frame"] / 1e3, "ns"],
        ["Temperature", m["temperature_K"], "K"], ["Contact radius", m["contact_radius_nm"], "nm"],
        ["Runtime", fmt_duration(m["runtime_s"]), ""],
    ], "Each frame is an independent relaxation from one Monte Carlo loop snapshot.", "molecular_dynamics")
    emit_table("Structure statistics", ["Quantity", "Mean", "Median", "Min", "Max", "Unit"], [
        describe_row("Radius of gyration", md.rg, "nm"), describe_row("Potential energy", md.epot, "kJ/mol"),
    ], "Radius of gyration measures compaction of the 3D structure.", "molecular_dynamics")
    map_figs.md_maps(md, ctx.geom)
    if ctx.replicas == 1:                                          # with replicas the frames of different runs are not one time series (see _replicas)
        map_figs.md_series(md)
    geo = map_figs.md_geometry(md, ctx.mc_ens, ctx.geom)
    emit_table("3D geometry", ["Quantity", "Value", "Unit"], [
        ["Scaling exponent nu (R ~ s^nu)", geo["nu"], "dimensionless"],
        ["Median loop-anchor distance", geo["loop_nm"], "nm"], ["Median distance, random pairs (same s)", geo["ctrl_nm"], "nm"],
        ["Loop compaction", geo["ctrl_nm"] / max(geo["loop_nm"], 1e-9), "x closer"],
    ], "nu = 1/3 compact globule, 1/2 ideal chain, ~0.59 swollen chain.", "molecular_dynamics")
    if ctx.mc_map is not None:
        map_figs.mc_vs_md(ctx.mc_map, md.avg_map, ctx.geom)
    map_feature_report(ctx, "md", md.avg_map, "3D structures (MD)")
    if ctx.features and "mc" in ctx.features:
        from loopsage.viz.map_figs import features_compare_figure
        from loopsage.validation.compare import experiment_features
        c = features_compare_figure(ctx.features["mc"], ctx.features["md"], ctx.geom, ctx.inputs.comp, experiment_features(ctx.loops_file, ctx.geom), ctx.mc_epi_profile)
        emit_table("MC vs MD map features", ["Feature", "Agreement", "Unit"], [
            *([["Compartments (MC epigenetic state vs MD PC1)", c["pc1"], "|Pearson r|"]] if c["pc1"] == c["pc1"] else []), ["Insulation score", c["insulation"], "Pearson r"],
            ["P(s) decay (log)", c["ps"], "Pearson r"], ["Boundaries within 2 beads", c["boundary_overlap"], "fraction of MC boundaries"],
        ], "How well the 3D structures reproduce the features of the loop ensemble.", "molecular_dynamics")
    if ctx.replicas > 1:
        _replicas(ctx)
        _per_replica(ctx)


if __name__ == "__main__":
    from loopsage.pipeline.runner import run_standalone
    run_standalone(NAME)
