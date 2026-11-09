"""Stage 3: 3D polymer molecular dynamics."""

from loopsage.common.genome import fmt_duration
from loopsage.common.utils import describe_row, emit_table
from loopsage.molecular_dynamics.engine import run_md
from loopsage.molecular_dynamics.engine import MDResult
from loopsage.reporting.report import REGISTRY
from loopsage.viz.map_figs import map_feature_report
from loopsage.viz import map_figs

NAME, ATTR, CACHE = "md", "md", "md.npz"


def enabled(cfg) -> bool:
    return bool(cfg.run_md)


def compute(ctx):
    return run_md(ctx.cfg, ctx.geom, ctx.mc_ens, ctx.dirs)


def load(path):
    return MDResult.load(path)


def describe(ctx):
    md, m = ctx.md, ctx.md.meta
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


if __name__ == "__main__":
    from loopsage.pipeline.runner import run_standalone
    run_standalone(NAME)
