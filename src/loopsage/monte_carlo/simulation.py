"""Monte Carlo output container (with save / load).

- Run the Monte Carlo sampler in chunks (progress bar); several replicas run in parallel threads.
"""
import json
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loopsage.common.runtime import n_threads, stage_bar
from loopsage.common.utils import Timer
from loopsage.logger import get_logger
from loopsage.monte_carlo.energy import total_energy
from loopsage.monte_carlo.sampler import build_params, init_state, lef_speeds, mc_block, seed_numba


# ---- Monte Carlo output container (with save / load) -----------------------

_ARRAYS = ("Ms", "Ns", "Es", "Ks", "Fs", "Bs", "ufs", "Ts", "epi", "acc")


@dataclass
class MCResult:
    Ms: np.ndarray    # (n_lef_total, n_samples) left-leg bead of each LEF over time
    Ns: np.ndarray    # right legs
    Es: np.ndarray    # total energy per snapshot (a.u.)
    Ks: np.ndarray    # crossing energy
    Fs: np.ndarray    # folding energy
    Bs: np.ndarray    # binding (data) energy
    ufs: np.ndarray   # unfolded fraction of the chain
    Ts: np.ndarray    # temperature at each snapshot
    epi: np.ndarray   # (n_beads, n_samples) epigenetic states
    acc: np.ndarray   # (3, 2) proposed/accepted for rebind, slide, epigenetic-state change
    meta: dict = field(default_factory=dict)   # n_beads, n_lef, mc_step, burnin, runtime_s, ...
    replicas: list = field(default_factory=list)   # the other independent chains (this object is replica 0)

    @property
    def n_samples(self) -> int:
        return self.Es.shape[0]

    @property
    def burnin_idx(self) -> int:
        """First post-burn-in snapshot."""
        return min(self.meta["burnin"] // self.meta["mc_step"], self.n_samples - 1)

    def production(self, arr):
        """Slice of the last axis after burn-in."""
        return arr[..., self.burnin_idx:]

    @property
    def chains(self) -> list:
        """All independent chains, this one first."""
        return [self] + list(self.replicas)

    def pooled(self) -> "MCResult":
        """All chains' post-burn-in snapshots joined in time (burn-in 0): the ensemble used for maps and MD."""
        if not self.replicas:
            return self
        cat = lambda k: np.concatenate([getattr(c, k)[..., c.burnin_idx:] for c in self.chains], axis=-1)
        meta = dict(self.meta, burnin=0, pooled_chains=len(self.chains))
        return MCResult(*[cat(k) for k in _ARRAYS[:-1]], np.sum([c.acc for c in self.chains], axis=0), meta)

    def save(self, path) -> None:
        """Write this chain to `path`; further chains go to mc_r1.npz, mc_r2.npz, ... beside it."""
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **{k: getattr(self, k) for k in _ARRAYS})
        for k, r in enumerate(self.replicas, start=1):
            r.save(Path(path).with_name(f"mc_r{k}.npz"))

    @classmethod
    def load(cls, path) -> "MCResult":
        z = np.load(path, allow_pickle=False)
        res = cls(**{k: z[k] for k in _ARRAYS}, meta=json.loads(str(z["meta"])))
        for k in range(1, res.meta.get("replicas", 1) if res.meta.get("replica", 0) == 0 else 1):
            res.replicas.append(cls.load(Path(path).with_name(f"mc_r{k}.npz")))
        return res


# ---- Run the Monte Carlo sampler in chunks (progress bar); several replicas run in parallel threads ----

log = get_logger(__name__)
_N_CHUNKS = 100  # progress-bar granularity (1 % each)


def _chain(cfg, p, n_tot, n_save, n_beads, seed, advance):
    """One independent chain with its own seed (numba's random state is per thread)."""
    fl = lambda: np.zeros(n_save)
    rec = (np.zeros((n_tot, n_save), np.int64), np.zeros((n_tot, n_save), np.int64),
           fl(), fl(), fl(), fl(), fl(), fl(), np.zeros((n_beads, n_save), np.int64))
    acc = np.zeros((3, 2), np.int64)
    seed_numba(seed)
    speeds = lef_speeds(cfg, n_tot, np.random.default_rng(seed))
    if p.use_speed:
        p = p._replace(speed_cdf=np.cumsum(speeds) / speeds.sum())
    ms, ns, S = init_state(p)
    E = total_energy(p, ms, ns, S)
    chunk = max(1, cfg.n_steps // _N_CHUNKS)
    for i0 in range(0, cfg.n_steps, chunk):
        i1 = min(i0 + chunk, cfg.n_steps)
        E = mc_block(p, i0, i1, cfg.n_steps, cfg.mc_step, ms, ns, S, E, rec, acc)
        advance(i1 - i0)
    return rec, acc, speeds


def run_mc(cfg, inputs, geom, quiet: bool = False) -> MCResult:
    """Sample LEF configurations; returns snapshots every cfg.mc_step steps (replica 0, others in .replicas). quiet = no progress bar or step messages."""
    say = log.debug if quiet else log.step
    n_lef = cfg.n_lef or max(inputs.loops.n_sites // 2, 1)
    p = build_params(cfg, inputs, geom.n_beads, n_lef)
    n_tot, n_save, n_rep = n_lef + cfg.n_lef2, -(-cfg.n_steps // cfg.mc_step), max(int(cfg.replicas), 1)
    workers = min(n_threads(cfg), n_rep)
    say("Sampling %d LEFs on %d beads: %d steps, %d snapshots, %d replica(s) on %d thread(s)",
             n_tot, geom.n_beads, cfg.n_steps, n_save, n_rep, workers)
    log.info("fold_norm=%.3g bind_norm=%.3g", p.fold_norm, p.bind_norm)
    log.info("LEF motion: %s, %s speeds", "random-walk sliding" if cfg.lef_rw else "outward extrusion", cfg.lef_speed_dist)
    base = cfg.seed if cfg.seed is not None else np.random.randint(0, 2**31 - 1 - n_rep)
    lock = threading.Lock()

    log.info("Compiling the sampler on first use (cached afterwards)")
    bar_ctx = nullcontext(lambda k: None) if quiet else stage_bar("Monte Carlo sampling" + (f" x{n_rep}" if n_rep > 1 else ""), cfg.n_steps * n_rep)
    with Timer() as t, bar_ctx as bar:
        def advance(k):
            with lock:
                bar(k)
        run = lambda r: _chain(cfg, p, n_tot, n_save, geom.n_beads, base + r, advance)
        if workers > 1:
            with ThreadPoolExecutor(workers) as ex:
                out = list(ex.map(run, range(n_rep)))
        else:
            out = [run(r) for r in range(n_rep)]

    meta = dict(n_beads=geom.n_beads, n_lef=n_lef, n_lef2=cfg.n_lef2, n_sites=inputs.loops.n_sites,
                n_steps=cfg.n_steps, mc_step=cfg.mc_step, burnin=cfg.burnin, method=cfg.method,
                fold_norm=p.fold_norm, bind_norm=p.bind_norm, k_norm=p.k_norm, runtime_s=t.seconds,
                epi_on=bool(p.epi_norm != 0.0), lef_speed_dist=cfg.lef_speed_dist, lef_rw=bool(cfg.lef_rw), lef_speed_spread=cfg.lef_speed_spread,
                lef_fast_fraction=cfg.lef_fast_fraction, lef_fast_ratio=cfg.lef_fast_ratio,
                replicas=n_rep, threads=workers, seed=int(base))
    chains = [MCResult(*rec[:7], rec[7], rec[8], acc, dict(meta, replica=r, seed=int(base) + r, lef_speed=v.tolist())) for r, (rec, acc, v) in enumerate(out)]
    chains[0].replicas = chains[1:]
    return chains[0]
