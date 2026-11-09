"""Stage 4: validation of the simulated contacts against the experiment."""

from loopsage.analysis.maps import pc1
from loopsage.common.utils import emit_table
from loopsage.logger import get_logger
from loopsage.reporting.report import REGISTRY
from loopsage.validation.compare import validate
from loopsage.viz.validation_figs import signal_figure, validation_summary

NAME, ATTR, CACHE = "analysis", "validation", None
log = get_logger(__name__)


def enabled(cfg) -> bool:
    return True


def _sources(ctx):
    """Simulated maps to validate: the loop ensemble (MC) and, when it ran, the 3D structures (MD)."""
    out = {"loops (MC)": ctx.mc_map}
    if ctx.md is not None:
        out["3D (MD)"] = ctx.md.avg_map
    return out


def compute(ctx):
    maps = _sources(ctx)
    ctx.val_source = list(maps)[-1]                                # headline source: MD when available
    out = {}
    for src, m in maps.items():
        if "MD" in src:                                            # MD: PC1 of the contact map
            track, name = (pc1(m, ctx.inputs.comp)[0] if ctx.inputs.comp is not None else None), "PC1 of the simulated map"
        else:                                                      # MC: average epigenetic state (skipped without the epigenetic model)
            track, name = ctx.mc_epi_profile, "average epigenetic state"
        out[src] = validate(m, ctx.cfg, ctx.geom, ctx.loops_file, ctx.inputs.comp, track, name)
    return out


def _hic(ctx):
    """Simulated maps against the optional Hi-C map: scorecard table, figures, headline facts."""
    from loopsage.validation.hic_compare import compare_hic, METRICS
    from loopsage.viz.hic_figs import hic_correlation_figure, hic_scores_figure, hic_validation_figure
    H, info = ctx.hic
    res = {src: compare_hic(m, H, state=ctx.mc_epi_profile, track=ctx.cfg.track_smooth) for src, m in _sources(ctx).items()}
    f = lambda v: "n/a" if v != v else f"{v:.2f}"
    emit_table("Validation vs Hi-C", ["Score", *res, "Meaning"],
               [[name, *[f(r[key]) for r in res.values()], note] for name, key, note in METRICS],
               f"Simulated contact maps against the Hi-C map ({info['resolution']:,} bp, {info['norm']}) on the same beads; 1 = identical, 0 = unrelated.", "validation")
    hic_validation_figure(res[ctx.val_source], ctx.geom, ctx.val_source)
    hic_correlation_figure(res[ctx.val_source], ctx.geom, ctx.val_source)
    hic_scores_figure(res)
    REGISTRY.fact("Hi-C O/E r", f"{res[ctx.val_source]['r_oe']:.2f}")
    return res


def describe(ctx):
    both = len(ctx.validation) > 1
    for src, res in ctx.validation.items():
        for n in res.notes:
            log.info(n)
        emit_table(f"Validation vs experiment (simulation: {src})",
                   ["Comparison", "Signal", "n", "Pearson r", "95% CI", "Spearman", "Kendall", "perm. p", "Strength"],
                   [[r["label"], r["signal"], r["n"], r["pearson"], f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]",
                     r["spearman"], r["kendall"], r["perm_p"] if r["perm_p"] is not None else "-", r["strength"]]
                    for r in res.rows],
                   "n = positions compared; correlations are dimensionless (-1 to 1).", "validation")
        if res.diagnostics and src == ctx.val_source:
            emit_table("Orientation diagnostics", ["Test", "n", "Result", "p-value", "Expected"],
                       [[d["name"], d["n"], d["value"], d["p"], d["expect"]] for d in res.diagnostics],
                       "Extra checks using CTCF orientation (where known).", "validation", level=1)
        for p in res.plots:
            signal_figure(p, ctx.geom, (("_" + ("md" if "MD" in src else "mc")) if both else ""), src if both else "")
    head = ctx.validation[ctx.val_source]
    if head.rows:
        REGISTRY.fact("Pearson r", f"{head.rows[0]['pearson']:.2f} ({head.rows[0]['strength']})")
        sm = [r for r in head.rows if r["label"] == "Primary (zero-excluded), smoothed"]
        if sm:
            REGISTRY.fact("Pearson r (smoothed)", f"{sm[0]['pearson']:.2f} ({sm[0]['strength']})")
    validation_summary({src: res.rows for src, res in ctx.validation.items()})
    if ctx.hic is not None:
        _hic(ctx)


if __name__ == "__main__":
    from loopsage.pipeline.runner import run_standalone
    run_standalone(NAME)
