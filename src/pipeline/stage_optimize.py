"""Stage (optional, --optimize): fit K, f, b with Nelder-Mead on short MC runs; the following MC stage uses the result."""
import json
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.runtime import fun_bar
from common.utils import emit_table
from logger import get_logger
from optimizer.nelder_mead import optimize, OptResult
from analysis.maps import compartment_from_hic
from optimizer.objective import loss_parts, loss_weights, Objective, target_map
from reporting.report import REGISTRY
from settings.loader import write_ini
from validation.compare import detect_format

NAME, ATTR, CACHE = "optimize", "optimization", "optimization.json"
log = get_logger(__name__)


def enabled(cfg) -> bool:
    return bool(cfg.optimize)


def save(res: OptResult, path) -> None:
    Path(path).write_text(json.dumps(res.__dict__, indent=1))


def load(path) -> OptResult:
    return OptResult(**json.loads(Path(path).read_text()))


def compute(ctx):
    cfg, g = ctx.cfg, ctx.geom
    hic = ctx.hic
    if hic is None and detect_format(ctx.loops_file) != "bedpe":
        log.warning("--optimize needs a BEDPE loop file or a Hi-C map (--hic) as the target: skipped.")
        return None
    if hic is not None:
        log.info("Optimizer target: Hi-C map (%s, %s bp, %s normalisation)", hic[1]["file"], f"{hic[1]['resolution']:,}", hic[1]["norm"])
    k0 = cfg.n_lef or max(ctx.inputs.loops.n_sites // 2, 1)
    log.info("Optimizer start: n_lef=%d f=%.3g b=%.3g, temperature %.3g -> %.3g; %d evaluations of %d MC steps", k0, cfg.folding_coeff, cfg.bind_coeff, cfg.t_init, cfg.t_final, cfg.opt_evals, cfg.opt_steps)
    log.info("Loss weights (eigenvectors, decay, insulation, loop size) for a %.3g Mb region: %s", g.length_bp / 1e6, ", ".join(f"{w:.2f}" for w in loss_weights(g.length_bp)))
    seed = (cfg.seed if cfg.seed is not None else 0) + 100_000                       # never the seed of the production run
    ref = compartment_from_hic(hic[0]) if hic is not None else ctx.inputs.comp          # A/B reference for the Potts model
    obj = Objective(cfg, ctx.inputs, g, target_map(cfg, ctx.inputs, g, ctx.loops_file, None if hic is None else hic[0]), cfg.opt_steps, seed, ref, sparse=hic is None)
    with fun_bar("Optimizer", cfg.opt_evals + 4) as tick:                             # search evaluations + 3 candidate checks + the start check
        res = optimize(obj, k0, cfg.folding_coeff, cfg.bind_coeff, g.n_beads, cfg.opt_evals, tick=tick,
                       patience=cfg.opt_patience if cfg.opt_early_stop else 0, min_gain=cfg.opt_min_gain,
                       log=lambda k, n, loss, best: log.info("evaluation %d/%d: loss %.4f (best %.4f)", k, n, loss, best))
    for d in (res.start, res.best):
        d["t_init"], d["t_final"] = cfg.t_init * d["T"], cfg.t_final * d["T"]
    log.step(verdict(res))
    return res


def verdict(res: OptResult) -> str:
    """One plain sentence: did the search converge, and did it beat the start?"""
    s, b = res.start, res.best
    how = ("converged (the simplex shrank below the tolerance)" if res.converged
           else f"stopped early after {res.evaluations} evaluations (no real improvement of the loss any more)" if res.stopped_early
           else f"did not converge (stopped at the budget of {res.evaluations} evaluations)")
    same = (s["K"], s["f"], s["b"], s["T"]) == (b["K"], b["f"], b["b"], b["T"])
    if same:
        return f"Optimizer {how}; no candidate beat the starting parameters on the long check, so they are kept (it gave up)."
    gain = 100 * (1 - b["loss"] / s["loss"]) if s["loss"] > 0 else 0.0
    tail = ("this is the optimum found" if res.converged else "the loss stopped improving, so this is a good point to stop (--no-opt-early-stop searches on)" if res.stopped_early
            else "just the best found so far, more evaluations (--opt-evals) may improve it")
    edge = [k for k, (lo, hi) in res.bounds.items() if b[k] <= lo * 1.001 or b[k] >= hi * 0.999]
    warn = f" Warning: {', '.join(edge)} sit at the edge of the search range, the optimum may lie beyond." if edge else ""
    return f"Optimizer {how}; loss {s['loss']:.3f} -> {b['loss']:.3f} ({gain:.0f}% lower): {tail}.{warn}"


def apply(ctx, res: OptResult) -> None:
    """Put the chosen K (first family only), f, b and temperatures into the run configuration and rewrite the resolved ini."""
    cfg, b = ctx.cfg, res.best
    cfg.n_lef, cfg.folding_coeff, cfg.bind_coeff, cfg.t_init, cfg.t_final = b["K"], b["f"], b["b"], b["t_init"], b["t_final"]
    for k in ("n_lef", "folding_coeff", "bind_coeff", "t_init", "t_final"):
        cfg.source[k] = "optimizer"
    write_ini(ctx.dirs.config, cfg.as_dict(), cfg.source)
    meta = ctx.dirs.data / "metadata"                                                 # small and important: always saved
    meta.mkdir(exist_ok=True)
    (meta / "optimized_parameters.json").write_text(json.dumps(dict(
        chosen=dict(n_lef=b["K"], folding_coeff=b["f"], bind_coeff=b["b"], t_init=b["t_init"], t_final=b["t_final"], t_scale=b["T"]),
        start=res.start, best=res.best, converged=res.converged, evaluations=res.evaluations, seconds=res.seconds, verdict=verdict(res)), indent=1))


def describe(ctx):
    res = ctx.optimization
    apply(ctx, res)
    s, b, tr = res.start, res.best, np.array(res.trace)
    gain = 100 * (1 - b["loss"] / s["loss"]) if s["loss"] > 0 else 0.0
    emit_table("Parameter optimizer (Nelder-Mead)", ["Parameter", "Start", "Optimized", "Unit"],
               [["Number of LEFs K (first family only)", s["K"], b["K"], "LEFs"], ["Folding coefficient f", s["f"], b["f"], ""],
                ["Binding coefficient b", s["b"], b["b"], ""], ["Initial temperature", s["t_init"], b["t_init"], "a.u."],
                ["Final temperature", s["t_final"], b["t_final"], "a.u. (Annealing)"], ["Loss (long check run)", s["loss"], b["loss"], "0 = perfect"],
                ["Evaluations", "", res.evaluations, "short MC runs"], ["Converged", "", "yes" if res.converged else "no (budget reached)", ""], ["Search time", "", res.seconds, "s"]],
               "Loss = weighted sum of the distance of the leading eigenvectors, the decay mismatch, the distance of the insulation profiles and the loop-size shortfall (weights set by the region size), against the " + (f"Hi-C map denoised over {ctx.cfg.hic_denoise:g} beads (decay, insulation and O/E taken from it). " if ctx.hic is not None
                                                                                                          else f"input loops drawn like the simulated ones (with communication between loops) and smoothed over {ctx.cfg.target_smooth:g} beads. ")
               + verdict(res) + " The Monte Carlo stage then runs all its steps with the optimized values.", "optimizer")
    W = loss_weights(ctx.geom.length_bp)
    emit_table("Optimizer loss terms (long check run)", ["Term", "Weight", "Start", "Optimized", "Meaning"],
               [["Eigenvector distance", W[0], s["parts"][0], b["parts"][0], "1 - cosine between the absolute values of the leading eigenvectors of the O/E correlation (0-1)"], ["Decay mismatch", W[1], s["parts"][1], b["parts"][1], "shape of P(s)"],
                ["Insulation distance", W[2], s["parts"][2], b["parts"][2], "mean difference of the insulation profiles scaled to 0-1 (domain boundaries)"],
                ["Loop-size shortfall", W[3], s["parts"][3], b["parts"][3], "mean simulated loop shorter than the mean input loop (log2)"], ["Total (weighted)", "", s["loss"], b["loss"], ""]],
               "Each term is 0 for a perfect match. The weights of the eigenvectors and of the insulation depend on the size of the region: the insulation dominates for TAD-scale regions, the eigenvectors for regions that span compartments, and they are equal in between. The loop-size term keeps the loops from collapsing to microscopic sizes when the input loops are sparse.", "optimizer")
    top = tr[np.argsort(tr[:, 1])[:8]]
    hit = [r[1] for r in tr if (int(r[2]), r[3], r[4], r[5]) == (b["K"], b["f"], b["b"], b["T"])]
    emit_table("Best optimizer evaluations", ["Evaluation", "K", "f", "b", "T scale", "Loss (short run)"],
               [[int(np.flatnonzero((tr == r).all(axis=1))[0]) + 1, int(r[2]), r[3], r[4], r[5], r[1]] for r in top],
               "Short-run losses are noisy: the chosen point scored %.3f in its search run and %.3f in the long check (the winner's curse)."
               % (hit[0] if hit else float("nan"), b["loss"]), "optimizer")
    REGISTRY.fact("Optimizer", ("converged" if res.converged else "not converged") + f": K={b['K']}, f={b['f']:.3g}, b={b['b']:.3g}, T x{b['T']:.2f}; loss {s['loss']:.3f} -> {b['loss']:.3f} ({gain:.0f}% lower)")
    from viz import optimizer_figs
    optimizer_figs.convergence_figure(res)
    optimizer_figs.landscape_figure(res)
    hic = ctx.hic
    args = (ctx.cfg, ctx.inputs, ctx.geom, ctx.loops_file, None if hic is None else hic[0])
    optimizer_figs.target_figure(target_map(*args, smooth=False), target_map(*args), ctx.geom, "Hi-C" if hic is not None else "Input loops")


if __name__ == "__main__":
    from pipeline.runner import run_standalone
    run_standalone(NAME)
