"""Run the Monte Carlo sampler in chunks so a progress bar can follow it."""
import numpy as np

from common.progress import stage_bar
from helpers.timing import Timer
from logger import get_logger
from monte_carlo.params import build_params
from monte_carlo.result import MCResult
from monte_carlo.sampler import init_state, mc_block, seed_numba
from monte_carlo.energy.total import total_energy

log = get_logger(__name__)
_N_CHUNKS = 100  # progress-bar granularity (1 % each)


def run_mc(cfg, inputs, geom) -> MCResult:
    """Sample LEF configurations; returns snapshots every cfg.mc_step steps."""
    n_lef = cfg.n_lef or max(inputs.loops.n_sites // 2, 1)
    p = build_params(cfg, inputs, geom.n_beads, n_lef)
    n_tot, n_save = n_lef + cfg.n_lef2, -(-cfg.n_steps // cfg.mc_step)
    log.info("%d LEFs on %d beads, %d snapshots; fold_norm=%.3g bind_norm=%.3g",
             n_tot, geom.n_beads, n_save, p.fold_norm, p.bind_norm)

    fl = lambda: np.zeros(n_save)
    rec = (np.zeros((n_tot, n_save), np.int64), np.zeros((n_tot, n_save), np.int64),
           fl(), fl(), fl(), fl(), fl(), fl(), np.zeros((geom.n_beads, n_save), np.int64))
    acc = np.zeros((3, 2), np.int64)

    seed_numba(cfg.seed if cfg.seed is not None else np.random.randint(0, 2**31 - 1))
    ms, ns, S = init_state(p)
    E = total_energy(p, ms, ns, S)
    chunk = max(1, cfg.n_steps // _N_CHUNKS)
    log.info("Compiling sampler on first use (cached afterwards)...")
    with Timer() as t, stage_bar("Monte Carlo sampling", cfg.n_steps) as advance:
        for i0 in range(0, cfg.n_steps, chunk):
            i1 = min(i0 + chunk, cfg.n_steps)
            E = mc_block(p, i0, i1, cfg.n_steps, cfg.mc_step, ms, ns, S, E, rec, acc)
            advance(i1 - i0)

    meta = dict(n_beads=geom.n_beads, n_lef=n_lef, n_lef2=cfg.n_lef2, n_sites=inputs.loops.n_sites,
                n_steps=cfg.n_steps, mc_step=cfg.mc_step, burnin=cfg.burnin, method=cfg.method,
                fold_norm=p.fold_norm, bind_norm=p.bind_norm, k_norm=p.k_norm, runtime_s=t.seconds,
                epi_on=bool(p.epi_norm != 0.0))
    return MCResult(*rec[:7], rec[7], rec[8], acc, meta)
