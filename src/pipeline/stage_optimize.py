"""Stage (optional, --optimize): fit K, f, b and the temperature with Nelder-Mead on short MC runs; the following MC stage uses the result.

With the Potts model on (epi_coeff > 0) and an A/B reference there are two stages: first the loop-extrusion parameters (Potts off; insulation, decay and loop size),
then epi_coeff and p_epigenetic (loop-extrusion parameters fixed; the full weighted loss, so PC1 against the average Potts state is balanced with the TAD terms by the region-size weights).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.genome import Geometry
from common.runtime import fun_bar
from common.utils import emit_table
from logger import get_logger
from optimizer.nelder_mead import ask_keep_best, optimize, optimize_potts, OptResult
from analysis.maps import compartment_from_hic
from preproc.loader import load_inputs
from optimizer.objective import loss_parts, loss_weights, Objective, PottsObjective, target_coverage, target_map
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


def pick_window(inputs, size: int) -> int:
    """First bead of the `size`-bead window with the most loop anchors."""
    a = np.asarray(inputs.loops.L, float) + np.asarray(inputs.loops.R, float)
    return int(np.argmax(np.convolve(a, np.ones(size), "valid")))


def rescale_K(res: OptResult, s: float) -> None:
    """Window -> full region: the number of LEFs scales with the length (the LEF density is what is fitted)."""
    for d in (res.start, res.best):
        d["K"] = max(int(round(d["K"] * s)), 1)
    res.bounds["K"] = [int(round(v * s)) for v in res.bounds["K"]]
    for row in res.trace:
        row[2] = row[2] * s


def compute(ctx):
    cfg, g = ctx.cfg, ctx.geom
    hic, inputs = ctx.hic, ctx.inputs
    s = 1.0                                                                          # window length / region length
    if g.n_beads > cfg.opt_max_beads:                                                # big region: the search runs on the densest window only
        size, bp = int(cfg.opt_max_beads), g.bp_per_bead
        i0 = pick_window(inputs, size)
        g = Geometry(g.chrom, g.start + i0 * bp, g.start + (i0 + size) * bp, size)
        inputs, s = load_inputs(cfg, g, ctx.loops_file), size / ctx.geom.n_beads
        hic = None if hic is None else (hic[0][i0:i0 + size, i0:i0 + size], hic[1])
        cfg = copy.copy(cfg)
        cfg.n_lef2 = int(round(cfg.n_lef2 * s))
        log.info("Large region (%d beads): the optimizer works on the %d-bead window %s with most loops and scales the number of LEFs back (--opt-max-beads).", ctx.geom.n_beads, size, g.label)
    if hic is None and detect_format(ctx.loops_file) != "bedpe":
        log.warning("--optimize needs a BEDPE loop file or a Hi-C map (--hic) as the target: skipped.")
        return None
    if hic is not None:
        log.info("Optimizer target: Hi-C map (%s, %s bp, %s normalisation)", hic[1]["file"], f"{hic[1]['resolution']:,}", hic[1]["norm"])
    k0 = max(int(round(ctx.cfg.n_lef * s)), 1) if ctx.cfg.n_lef else max(inputs.loops.n_sites // 2, 1)
    if cfg.opt_interactive and sys.stdin.isatty():
        log.info("Press Enter at any time to see the best parameters so far and choose whether to keep them and stop the search.")
    log.info("Optimizer start: n_lef=%d f=%.3g b=%.3g, temperature %.3g -> %.3g; %d evaluations of %d MC steps", k0, cfg.folding_coeff, cfg.bind_coeff, cfg.t_init, cfg.t_final, cfg.opt_evals, cfg.opt_steps)
    log.info("Loss weights (eigenvectors, decay, insulation, loop size) for a %.3g Mb region: %s", g.length_bp / 1e6, ", ".join(f"{w:.2f}" for w in loss_weights(g.length_bp)))
    seed = (cfg.seed if cfg.seed is not None else 0) + 100_000                       # never the seed of the production run
    ref = compartment_from_hic(hic[0], smooth=cfg.track_smooth) if hic is not None else inputs.comp          # A/B reference for the Potts model
    two = float(cfg.epi_coeff) > 0 and ref is not None                                # Potts on with an A/B reference: two stages
    if float(cfg.epi_coeff) > 0 and ref is None:
        log.warning("Potts model on but no A/B reference (no compartment track or Hi-C map): epi_coeff is not optimized.")
    if two:
        log.info("Potts model on: stage 1 fits the loop-extrusion parameters (insulation, decay, loop size), stage 2 fits epi_coeff and p_epigenetic (PC1 vs <s> balanced with the TAD terms).")
    target = target_map(cfg, inputs, g, ctx.loops_file, None if hic is None else hic[0])
    cover = target_coverage(g, ctx.loops_file) if hic is None else None
    make = lambda cls, c, sd, **kw: cls(c, inputs, g, target, cfg.opt_steps, sd, ref, sparse=hic is None, cover=cover, **kw)
    obj = make(Objective, cfg, seed, loops_only=two)
    with fun_bar("Optimizer", cfg.opt_evals + 4) as tick:                             # search evaluations + 3 candidate checks + the start check
        res = optimize(obj, k0, cfg.folding_coeff, cfg.bind_coeff, g.n_beads, cfg.opt_evals, tick=tick,
                       patience=cfg.opt_patience if cfg.opt_early_stop else 0, min_gain=cfg.opt_min_gain,
                       ask=ask_keep_best(["K", "f", "b", "T scale"]) if cfg.opt_interactive else None,
                       log=lambda k, n, loss, best: log.info("evaluation %d/%d: loss %.4f (best %.4f)", k, n, loss, best))
    for d in (res.start, res.best):
        d["t_init"], d["t_final"] = cfg.t_init * d["T"], cfg.t_final * d["T"]
    log.step(verdict(res))
    if two:
        res.potts = potts_stage(ctx, res, lambda c: make(PottsObjective, c, seed + 50_000), s)
    if s < 1:
        rescale_K(res, 1 / s)
    return res


def potts_stage(ctx, res, make, s: float = 1.0) -> dict:
    """Stage 2: epi_coeff and p_epigenetic, with the loop-extrusion parameters of stage 1 fixed; the full weighted loss balances PC1 with the TAD terms."""
    cfg, b = ctx.cfg, res.best
    c = copy.copy(cfg)
    c.n_lef, c.folding_coeff, c.bind_coeff, c.t_init, c.t_final = b["K"], b["f"], b["b"], b["t_init"], b["t_final"]
    c.n_lef2 = int(round(cfg.n_lef2 * s))
    evals = max(10, cfg.opt_evals // 3)
    log.info("Potts stage: start epi_coeff=%.3g p_epigenetic=%.3g; %d evaluations of %d MC steps", cfg.epi_coeff, cfg.p_epigenetic, evals, cfg.opt_steps)
    with fun_bar("Optimizer (Potts)", evals + 4) as tick:
        out = optimize_potts(make(c), cfg.epi_coeff, cfg.p_epigenetic, evals, tick=tick,
                             patience=cfg.opt_patience if cfg.opt_early_stop else 0, min_gain=cfg.opt_min_gain,
                             ask=ask_keep_best(["epi_coeff", "p_epigenetic"]) if cfg.opt_interactive else None,
                             log=lambda k, n, loss, best: log.info("Potts evaluation %d/%d: loss %.4f (best %.4f)", k, n, loss, best))
    s, o = out["start"], out["best"]
    log.step(f"Potts stage: epi_coeff {s['epi']:.3g} -> {o['epi']:.3g}, p_epigenetic {s['p']:.3g} -> {o['p']:.3g}; loss {s['loss']:.3f} -> {o['loss']:.3f}.")
    return out


def verdict(res: OptResult) -> str:
    """One plain sentence: did the search converge, and did it beat the start?"""
    s, b = res.start, res.best
    how = ("stopped by you (the best parameters so far were kept)" if res.by_user else "converged (the simplex shrank below the tolerance)" if res.converged
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
    keys = ["n_lef", "folding_coeff", "bind_coeff", "t_init", "t_final"]
    if res.potts:
        cfg.epi_coeff, cfg.p_epigenetic = res.potts["best"]["epi"], res.potts["best"]["p"]
        keys += ["epi_coeff", "p_epigenetic"]
    for k in keys:
        cfg.source[k] = "optimizer"
    write_ini(ctx.dirs.config, cfg.as_dict(), cfg.source)
    meta = ctx.dirs.data / "metadata"                                                 # small and important: always saved
    meta.mkdir(exist_ok=True)
    (meta / "optimized_parameters.json").write_text(json.dumps(dict(
        chosen=dict(n_lef=b["K"], folding_coeff=b["f"], bind_coeff=b["b"], t_init=b["t_init"], t_final=b["t_final"], t_scale=b["T"]),
        start=res.start, best=res.best, potts=None if not res.potts else {k: v for k, v in res.potts.items() if k != "trace"}, converged=res.converged, evaluations=res.evaluations, seconds=res.seconds, verdict=verdict(res)), indent=1))


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
               "Loss = weighted sum of the distance of the leading eigenvectors, the decay mismatch, the distance of the insulation profiles and the loop-size mismatch (weights set by the region size), against the " + (f"Hi-C map denoised over {ctx.cfg.hic_denoise:g} beads (decay, insulation and O/E taken from it). " if ctx.hic is not None
                                                                                                          else f"input loops drawn like the simulated ones (with communication between loops) and smoothed over {ctx.cfg.target_smooth:g} beads. ")
               + verdict(res) + " The Monte Carlo stage then runs all its steps with the optimized values.", "optimizer")
    W = loss_weights(ctx.geom.length_bp)
    if res.potts:
        W = (0.0, *W[1:])                                                            # stage 1 fits only the loop-extrusion terms
        ps, pb = res.potts["start"], res.potts["best"]
        emit_table("Potts-model optimizer (stage 2)", ["Parameter", "Start", "Optimized", "Range"],
                   [["Coupling epi_coeff", ps["epi"], pb["epi"], "%.3g - %.3g" % tuple(res.potts["bounds"]["epi"])], ["State-change probability p_epigenetic", ps["p"], pb["p"], "%.2g - %.2g" % tuple(res.potts["bounds"]["p"])],
                    ["Loss (weighted, all terms)", ps["loss"], pb["loss"], "0 = perfect"], ["Evaluations", "", res.potts["evaluations"], "short MC runs"], ["Search time", "", res.potts["seconds"], "s"]],
                   "With the Potts model on the search has two stages. Stage 1 (table above) fits K, f, b and the temperature on the insulation, the decay and the loop size only, with the Potts model off. "
                   "Stage 2 keeps them fixed and fits epi_coeff and p_epigenetic with the full loss, where the average Potts state of each bead is compared with the A/B reference (PC1 of the Hi-C map or the compartment track) with the region-size weight: the coupling is balanced between the compartments and the TAD terms, and it matters little in a small region.", "optimizer")
    emit_table("Optimizer loss terms of the loop-extrusion stage (long check run)" if res.potts else "Optimizer loss terms (long check run)", ["Term", "Weight", "Start", "Optimized", "Meaning"],
               [["Eigenvector distance", W[0], s["parts"][0], b["parts"][0], "1 - cosine between the absolute values of the leading eigenvectors of the O/E correlation (0-1)"], ["Decay mismatch", W[1], s["parts"][1], b["parts"][1], "shape of P(s)"],
                ["Insulation distance", W[2], s["parts"][2], b["parts"][2], "mean difference of the insulation profiles scaled to 0-1 (domain boundaries)"],
                ["Loop-size mismatch", W[3], s["parts"][3], b["parts"][3], "median simulated loop differs from the median input loop by more than a factor 2 (log2)"], ["Total (weighted)", "", s["loss"], b["loss"], ""]],
               "Each term is 0 for a perfect match. The weights of the eigenvectors and of the insulation depend on the size of the region: the insulation dominates for TAD-scale regions, the eigenvectors for regions that span compartments, and they are equal in between (with the Potts model on the eigenvector weight is 0 here: PC1 is fitted in stage 2). The loop-size term keeps the loops from collapsing to microscopic sizes when the input loops are sparse.", "optimizer")
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
