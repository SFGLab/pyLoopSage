"""Bundle of everything the (numba) sampler needs, plus normalisation constants."""
from collections import namedtuple

import numpy as np

# Energy terms are scaled so coefficients (f, b, kappa) are comparable across systems.
MCParams = namedtuple("MCParams", [
    "N_beads", "N_lef", "N_lef2",
    "fold_norm", "fold_norm2", "bind_norm", "k_norm", "epi_norm",
    "cross_loop", "between", "loss_mode", "p_spin", "n_epi", "step_length", "rw",
    "annealing", "T", "T_min",
    "L", "R", "J", "J_loss", "h", "BWs", "r", "use_track", "cdf",
])


def build_params(cfg, inputs, n_beads: int, n_lef: int) -> MCParams:
    """Assemble MCParams from config + preprocessed inputs."""
    lo, n_tot = inputs.loops, n_lef + cfg.n_lef2
    if n_tot >= n_beads:
        raise ValueError(f"{n_tot} LEFs do not fit on {n_beads} beads; reduce n_lef or raise n_beads")
    if lo.n_sites == 0:
        raise ValueError("no anchors found in the region")

    fold = -n_beads / (n_tot * np.log(n_beads / n_tot))
    n_bw = inputs.BWs.shape[0]
    r = np.array(cfg.bw_strengths, float) if cfg.bw_strengths else np.full(n_bw, -n_beads / 10)
    track = inputs.lef_track
    step = int(lo.loop_lengths.min()) if cfg.variable_sliding_step and len(lo.loop_lengths) else 1
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    h = inputs.comp if inputs.comp is not None else np.zeros(n_beads)

    return MCParams(
        n_beads, n_lef, cfg.n_lef2,
        fold * cfg.folding_coeff, fold * cfg.folding_coeff2, -2 * n_beads * cfg.bind_coeff / lo.n_sites,
        cfg.cross_coeff * 1e4, float(cfg.epi_coeff),
        bool(cfg.cross_loop), bool(cfg.between_families_penalty), int(cfg.data_loss_mode),
        float(cfg.p_spin), int(cfg.n_epi_states), max(step, 1), bool(cfg.lef_rw),
        cfg.method == "Annealing", float(cfg.t_init), float(cfg.t_final),
        f64(lo.L), f64(lo.R), f64(lo.J), f64(lo.J_loss), f64(h), f64(inputs.BWs).reshape(n_bw, n_beads), f64(r),
        track is not None, f64(np.cumsum(track / track.sum())) if track is not None else np.zeros(0),
    )
