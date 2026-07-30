"""
loopsage_tuner.py
=================
Automatic hyperparameter search for LoopSage stochastic simulations.

Why not gradient descent / coordinate descent?
-----------------------------------------------
The loss is STOCHASTIC — two identical runs give slightly different losses
due to MC randomness.  Coordinate descent wastes most evaluations sweeping
bad axes, and can be fooled by noise into premature stopping.

Algorithm: CMA-ES  (Covariance Matrix Adaptation – Evolution Strategy)
-----------------------------------------------------------------------
CMA-ES is the standard algorithm for noisy, black-box, continuous
optimisation.  It maintains a Gaussian search distribution N(m, sigma^2 C)
over parameter space and updates it each generation by:
  1. Sampling lambda candidate solutions
  2. Ranking them by (averaged) loss
  3. Moving the mean m toward the better half
  4. Adapting the covariance C to learn correlations between parameters
  5. Adapting the global step size sigma via a cumulative path length criterion

Key advantages over coordinate descent / simulated annealing:
  - Learns parameter correlations (e.g. f and N_lef are coupled)
  - Self-adapts step sizes per axis — no manual grid needed
  - Much faster convergence: typically far fewer evaluations than grid search
  - Handles noisy objectives naturally via ranking (not loss values directly)
  - Population-based: evaluates lambda points per generation

Pipeline
--------
Stage 0 – Warm start
  Sample n_warm random points; use the best 30% centroid as CMA initial mean.

Stage 1 – CMA-ES
  Run for up to max_generations.  Each generation:
    a) draw lambda candidates in continuous space
    b) snap each to the nearest grid point
    c) evaluate (cache avoids re-running identical grid points)
    d) rank and update CMA state
  Early stopping: if best-loss EMA has not improved by min_delta for
  patience generations AND inter-generation std < loss_std_tol.

Stage 2 – Neighbourhood polish
  Check all +-1 grid neighbours of the CMA best, averaged over
  polish_n_repeats runs.  Repeat until no improvement.

Parameters tuned (kappa is NOT tuned — set it large externally)
---------------------------------------------------------------
  T      in [1.0, 3.0]
  f      in [0.5, 4.0]
  b      in [0.5, 4.0]
  N_lef  in [N_CTCF//2, N_CTCF*2]

Loss — redesigned to prevent large-f/b cheating
------------------------------------------------
  L = w_dist    * dist_loss
    + w_density * density_loss
    + w_lendist * loop_len_loss
    + w_unfold  * unfold_loss
    + w_comp    * comp_loss  [optional]

  dist_loss:
    Simulated distance matrix = 1/(contact+eps).
    Reference distance matrix derived from J_loss (smoothed).
    Pearson on upper triangle.
    Catches large-f cheat: uniform extrusion -> flat distance matrix -> bad score.

  density_loss:
    1D contact marginal (how often each bead is contacted) vs
    normalised CTCF binding profile L+R.
    Pearson.  Catches flat over-extrusion.

  loop_len_loss:
    KL divergence between simulated and empirical loop-length distributions.
    Penalises wrong loop sizes.

  unfold_loss:
    Mean fraction of beads not covered by any loop (from ufs trajectory).
    Penalises tiny collapsed loops.

  comp_loss [optional]:
    |Pearson(mean_spin[burnin:], h_ref)|
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import io
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import pearsonr, entropy


# ---------------------------------------------------------------------------
# stdout suppressor  (silences numba "Progress: X%" and OpenMM prints)
# ---------------------------------------------------------------------------
class _Silence:
    """
    Redirects only stdout to /dev/null during a simulation run.

    The tuner's own messages use logging (which writes to stderr via
    StreamHandler), so they remain visible.  Only the simulation's
    print()-based progress output (numba JIT prints, OpenMM reporter)
    is silenced.
    """
    def __enter__(self):
        self._old_stdout = sys.stdout
        self._devnull    = open(os.devnull, "w")
        sys.stdout       = self._devnull
        return self

    def __exit__(self, *args):
        sys.stdout = self._old_stdout
        self._devnull.close()

# ---------------------------------------------------------------------------
# Logging + lightweight progress bar (works without tqdm)
# ---------------------------------------------------------------------------
log = logging.getLogger("loopsage_tuner")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[tuner] %(levelname)s %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _progress(iterable, desc="", total=None):
    """tqdm if available, otherwise a minimal ASCII bar."""
    try:
        from tqdm import tqdm
        yield from tqdm(iterable, desc=desc, total=total, ncols=80)
        return
    except ImportError:
        pass
    items = list(iterable)
    n = total or len(items)
    for i, item in enumerate(items):
        pct = int(100 * (i + 1) / max(n, 1))
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\r{desc} [{bar}] {pct:3d}% ({i+1}/{n})", end="", flush=True)
        yield item
    print()


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def _frange(start: float, stop: float, step: float) -> List[float]:
    vals, v = [], start
    while v <= stop + step * 1e-9:
        vals.append(round(v, 10))
        v += step
    return vals


def default_grids(N_CTCF: int) -> Dict[str, list]:
    """
    Default search grids (kappa excluded — fix it externally).

      T     : 1.0 to 3.0, step 0.2  (11 values)
      f     : 0.5 to 4.0, step 0.2  (18 values)
      b     : 0.5 to 4.0, step 0.2  (18 values)
      N_lef : N_CTCF//2 to N_CTCF*2 (approx 10 values)
    """
    lef_step = max(1, N_CTCF // 10)
    return {
        "T":     _frange(1.0, 3.0, 0.2),
        "f":     _frange(0.5, 4.0, 0.2),
        "b":     _frange(0.5, 4.0, 0.2),
        "N_lef": list(range(max(1, N_CTCF // 2), N_CTCF * 2 + 1, lef_step)),
    }


def _snap(val, grid: list):
    arr = np.asarray(grid, dtype=float)
    return grid[int(np.argmin(np.abs(arr - float(val))))]


def _grid_idx(val, grid: list) -> int:
    arr = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(arr - float(val))))


def _to_unit(params: dict, grids: Dict[str, list]) -> np.ndarray:
    """Map grid-snapped params to [0,1]^n (sorted key order)."""
    keys = sorted(grids.keys())
    return np.array([_grid_idx(params[k], grids[k]) / max(len(grids[k]) - 1, 1)
                     for k in keys])


def _from_unit(x: np.ndarray, grids: Dict[str, list]) -> dict:
    """Map [0,1]^n back to nearest grid values (sorted key order)."""
    keys = sorted(grids.keys())
    out  = {}
    for i, k in enumerate(keys):
        g   = grids[k]
        idx = int(round(float(np.clip(x[i], 0, 1)) * (len(g) - 1)))
        out[k] = g[np.clip(idx, 0, len(g) - 1)]
    return out


# ---------------------------------------------------------------------------
# TunerConfig
# ---------------------------------------------------------------------------
@dataclass
class TunerConfig:
    """
    Full configuration for the three-stage tuner.

    CMA-ES
    ------
    lam              : population size per generation (lambda).
    max_generations  : hard cap on CMA generations.
    sigma0           : initial step size in normalised [0,1] space.
    n_repeats        : independent short runs averaged per candidate.
    n_warm           : random warm-start samples before CMA.

    Early stopping
    --------------
    patience         : generations with no EMA improvement before checking std.
    min_delta        : minimum EMA improvement to reset patience.
    loss_std_tol     : stop when inter-generation EMA std < this.
    ema_alpha        : EMA smoothing factor for generation best-losses.

    Polish
    ------
    n_polish_rounds  : max neighbourhood polish rounds after CMA.
    polish_n_repeats : averaged evaluations per neighbour.

    Loss weights
    ------------
    w_dist     : contact-distance heatmap Pearson   (main anti-cheat signal)
    w_density  : 1D loop-density vs CTCF profile    (prevents flat over-extrusion)
    w_lendist  : loop-length KL divergence          (prevents wrong-size loops)
    w_unfold   : unfolding fraction penalty
    w_comp     : compartment agreement [optional, 0 to disable]

    Simulation
    ----------
    N_steps_short    : MC steps per evaluation run.
    MC_step          : sampling frequency.
    burnin           : burn-in steps discarded from loss.
    T_min_fraction   : T_min = T * T_min_fraction (Annealing mode only).
    mode             : 'Annealing' or 'Metropolis'.
    kappa            : fixed crossing coefficient (NOT tuned).
    grids            : per-parameter grids. None -> default_grids(N_CTCF).
    dist_smooth_sigma: Gaussian sigma for reference distance map smoothing.
    n_lendist_bins   : histogram bins for loop-length KL.
    verbose          : log every generation.
    """
    # CMA-ES
    lam:               int   = 10
    max_generations:   int   = 60
    sigma0:            float = 0.3
    n_repeats:         int   = 1
    n_warm:            int   = 20

    # Early stopping
    patience:          int   = 8
    min_delta:         float = 5e-4
    loss_std_tol:      float = 3e-3
    ema_alpha:         float = 0.3

    # Polish
    n_polish_rounds:   int   = 4
    polish_n_repeats:  int   = 2

    # Loss weights
    w_dist:            float = 1.0
    w_density:         float = 0.5
    w_lendist:         float = 0.3
    w_unfold:          float = 0.2
    w_comp:            float = 0.4

    # Simulation
    N_steps_short:     int   = 10_000
    MC_step:           int   = 500
    burnin:            int   = 2_000
    T_min_fraction:    float = 0.3
    mode:              str   = "Annealing"
    kappa:             float = 1.0
    grids:             Optional[Dict[str, list]] = None
    dist_smooth_sigma: float = 3.0
    n_lendist_bins:    int   = 15
    verbose:           bool  = True


# ---------------------------------------------------------------------------
# Reference data (computed once from preprocessed sim, before any evaluations)
# ---------------------------------------------------------------------------
class _ReferenceData:
    """
    Pre-computes all reference signals from sim.J_loss, sim.L, sim.R, sim.J.
    Zero simulations needed here.
    """

    def __init__(self, sim, config: TunerConfig):
        N   = sim.N_beads
        cfg = config

        # ── reference distance matrix ────────────────────────────────
        Jraw = sim.J_loss.copy()
        Jraw = (Jraw - Jraw.min()) / (Jraw.max() - Jraw.min() + 1e-9)
        ref_dist = 1.0 / (Jraw + 0.05)

        # polymer backbone prior: log-distance in sequence
        seq_dist = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
        seq_dist = np.log1p(seq_dist.astype(float)) + 1.0
        ref_dist = ref_dist * seq_dist
        ref_dist = (ref_dist + ref_dist.T) / 2
        ref_dist = gaussian_filter(ref_dist, sigma=cfg.dist_smooth_sigma)
        np.fill_diagonal(ref_dist, 0.0)
        self.ref_dist = ref_dist

        # upper-triangle mask (k=3 excludes near-diagonal noise)
        self.tri_i, self.tri_j = np.triu_indices(N, k=3)

        # ── 1-D CTCF binding density ─────────────────────────────────
        lr = sim.L + sim.R
        lr = gaussian_filter1d(lr, sigma=max(1.0, N / 50))
        lr = lr / (lr.sum() + 1e-9)
        self.ctcf_density = lr

        # ── empirical loop-length histogram ──────────────────────────
        stats = getattr(sim, "loop_stats", {})
        ll    = stats.get("loop_length", {})
        self.target_loop_mean = ll.get("mean", None)

        Ji, Jj = np.where(sim.J > 0)
        lengths = np.abs(Ji.astype(int) - Jj.astype(int))
        lengths = lengths[lengths > 1]
        self.len_bins = np.linspace(0, N, cfg.n_lendist_bins + 1)
        if len(lengths) > 0:
            hist, _ = np.histogram(lengths, bins=self.len_bins, density=False)
            hist    = hist.astype(float) + 1.0     # Laplace smoothing
            self.ref_len_hist = hist / hist.sum()
        else:
            self.ref_len_hist = None

        # ── compartment field ────────────────────────────────────────
        self.h_ref   = sim.h
        self.has_epi = (sim.h is not None)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
def _sim_contact_map(Ms: np.ndarray, Ns: np.ndarray,
                     N_beads: int, burnin_idx: int) -> np.ndarray:
    heat   = np.zeros((N_beads, N_beads), dtype=np.float64)
    frames = Ms.shape[1]
    for t in range(burnin_idx, frames):
        for k in range(Ms.shape[0]):
            m, n = int(Ms[k, t]), int(Ns[k, t])
            if 0 <= m < N_beads and 0 <= n < N_beads and m != n:
                heat[m, n] += 1.0
                heat[n, m] += 1.0
    return heat / max(frames - burnin_idx, 1)


def _pearson_safe(a, b) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(r) if np.isfinite(r) else 0.0


def _kl_safe(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float) + 1e-9
    q = np.asarray(q, float) + 1e-9
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------
class _Evaluator:
    """
    Runs short LoopSage simulations and returns the scalar loss.

    Speed features
    --------------
    - Result cache: identical grid-snapped parameter combinations are
      never re-simulated.  Cache hit rate is reported at the end.
    - Surrogate pre-screening: before the full simulation, a cheap
      quadratic surrogate (fitted on all previously evaluated points)
      predicts the loss.  If the prediction is worse than
      (best_loss + surrogate_reject_margin), the candidate is skipped.
      This saves ~30-50% of simulations in later CMA-ES generations.
    - Per-run wall-clock timing logged for budget tracking.
    """

    def __init__(self, sim, config: TunerConfig, ref: _ReferenceData):
        self.sim        = sim
        self.cfg        = config
        self.ref        = ref
        self.n_calls    = 0    # actual simulation runs
        self.n_cache    = 0    # cache hits
        self.n_skipped  = 0    # surrogate-rejected
        self.best_loss  = np.inf
        self._t_start   = time.time()
        self.N_beads    = sim.N_beads
        self.burnin_idx = max(1, config.burnin // config.MC_step)
        self._cache: Dict[tuple, List[float]] = {}
        # surrogate data: list of (x_unit, loss) for quadratic fitting
        self._surr_X: List[np.ndarray] = []
        self._surr_y: List[float]      = []

    def evaluate(self, params: dict, n_repeats: int = 1,
                 use_surrogate: bool = False,
                 surrogate_reject_margin: float = 0.05) -> float:
        key    = self._key(params)
        cached = self._cache.get(key, [])

        # ── cache hit ──────────────────────────────────────────────
        if len(cached) >= n_repeats:
            self.n_cache += 1
            return float(np.mean(cached[:n_repeats]))

        # ── surrogate pre-screen ───────────────────────────────────
        if use_surrogate and len(self._surr_X) >= 6 and np.isfinite(self.best_loss):
            pred = self._surrogate_predict(params)
            threshold = self.best_loss + surrogate_reject_margin
            if pred is not None and pred > threshold:
                self.n_skipped += 1
                # return surrogate prediction so CMA ranking still works
                return float(pred)

        # ── real simulations ───────────────────────────────────────
        needed = max(0, n_repeats - len(cached))
        for _ in range(needed):
            cached.append(self._single_run(params))
        self._cache[key] = cached

        result = float(np.mean(cached[:n_repeats]))
        if result < self.best_loss:
            self.best_loss = result

        # feed surrogate
        x_unit = _to_unit(params, self.cfg.grids)
        self._surr_X.append(x_unit)
        self._surr_y.append(result)

        return result

    def __call__(self, params: dict) -> float:
        return self.evaluate(params, n_repeats=self.cfg.n_repeats)

    def stats_str(self) -> str:
        total   = self.n_calls + self.n_cache + self.n_skipped
        elapsed = time.time() - self._t_start
        return (f"sims={self.n_calls}  cache_hits={self.n_cache}  "
                f"surrogate_skipped={self.n_skipped}  "
                f"total_queries={total}  elapsed={elapsed/60:.1f}min")

    def _surrogate_predict(self, params: dict) -> Optional[float]:
        """
        Fit a diagonal quadratic (ridge-regularised) on all seen points
        and predict the loss for `params`.  Returns None if fitting fails.
        """
        X = np.array(self._surr_X)   # (n, d)
        y = np.array(self._surr_y)
        x = _to_unit(params, self.cfg.grids)

        # feature vector: [x, x^2]  (diagonal quadratic, no cross terms)
        def featurise(v):
            return np.concatenate([v, v ** 2])

        Phi = np.array([featurise(xi) for xi in X])
        phi = featurise(x)

        try:
            lam_reg = 1e-3
            A = Phi.T @ Phi + lam_reg * np.eye(Phi.shape[1])
            b_vec = Phi.T @ y
            w = np.linalg.solve(A, b_vec)
            return float(phi @ w)
        except np.linalg.LinAlgError:
            return None

    def _single_run(self, params: dict) -> float:
        self.n_calls += 1
        cfg   = self.cfg
        T     = float(params["T"])
        f     = float(params["f"])
        b     = float(params["b"])
        N_lef = int(params["N_lef"])
        T_min = (T * cfg.T_min_fraction) if cfg.mode == "Annealing" else T

        t0 = time.time()
        log.info("    ┌─ run #%d │ T=%.2f  T_min=%.2f  f=%.2f  b=%.2f  "
                 "N_lef=%d  kappa=%.2f",
                 self.n_calls, T, T_min, f, b, N_lef, cfg.kappa)

        orig           = self.sim.N_lef
        self.sim.N_lef = N_lef
        try:
            with _Silence():
                _, Ms, Ns, _, _, _, ufs, epi = self.sim.run_energy_minimization(
                    N_steps = cfg.N_steps_short,
                    MC_step = cfg.MC_step,
                    burnin  = cfg.burnin,
                    T       = T,
                    T_min   = T_min,
                    mode    = cfg.mode,
                    viz     = False,
                    save    = False,
                    f       = f,
                    b       = b,
                    kappa   = cfg.kappa,
                )
        except Exception as e:
            log.warning("    └─ FAILED: %s", e)
            return 1e6
        finally:
            self.sim.N_lef = orig

        loss    = self._loss(Ms, Ns, ufs, epi)
        elapsed = time.time() - t0
        log.info("    └─ done in %.1fs │ loss=%.4f │ best_so_far=%.4f",
                 elapsed, loss, self.best_loss)
        return loss

    def _loss(self, Ms, Ns, ufs, epi) -> float:
        cfg = self.cfg
        ref = self.ref
        bi  = self.burnin_idx
        N   = self.N_beads
        L   = 0.0

        contact = _sim_contact_map(Ms, Ns, N, bi)

        terms = {}   # name -> weighted contribution, for the log summary

        # 1. distance heatmap
        if cfg.w_dist > 0:
            sim_dist = 1.0 / (contact + 0.05)
            np.fill_diagonal(sim_dist, 0.0)
            r = _pearson_safe(sim_dist[ref.tri_i, ref.tri_j],
                              ref.ref_dist[ref.tri_i, ref.tri_j])
            contrib = cfg.w_dist * (1.0 - r)
            L += contrib
            terms["dist(r=%.3f)" % r] = contrib

        # 2. 1D loop-density vs CTCF profile
        if cfg.w_density > 0:
            density = contact.sum(axis=1)
            total   = density.sum()
            if total > 0:
                density /= total
            r = _pearson_safe(density, ref.ctcf_density)
            contrib = cfg.w_density * (1.0 - r)
            L += contrib
            terms["density(r=%.3f)" % r] = contrib

        # 3. loop-length KL divergence
        if cfg.w_lendist > 0 and ref.ref_len_hist is not None:
            spans = (Ns - Ms).astype(float)[:, bi:]
            if spans.size > 0:
                sim_hist, _ = np.histogram(spans.ravel(),
                                           bins=ref.len_bins, density=False)
                sim_hist    = sim_hist.astype(float) + 1.0
                sim_hist   /= sim_hist.sum()
                mean_sim_len = float(np.average(
                    0.5 * (ref.len_bins[:-1] + ref.len_bins[1:]), weights=sim_hist))
                kl      = min(_kl_safe(ref.ref_len_hist, sim_hist), 3.0)
                contrib = cfg.w_lendist * kl
                L += contrib
                terms["lendist(KL=%.3f,mean=%.1f)" % (kl, mean_sim_len)] = contrib

        # 4. unfolding penalty
        if cfg.w_unfold > 0 and len(ufs) > bi:
            uf_val  = float(np.mean(ufs[bi:]))
            contrib = cfg.w_unfold * uf_val
            L += contrib
            terms["unfold(%.3f)" % uf_val] = contrib

        # 5. compartment agreement
        if cfg.w_comp > 0 and ref.has_epi and epi.shape[1] > bi:
            mean_spin = epi[:, bi:].mean(axis=1)
            r = _pearson_safe(mean_spin, ref.h_ref)
            contrib = cfg.w_comp * (1.0 - abs(r))
            L += contrib
            terms["comp(r=%.3f)" % r] = contrib

        # per-run breakdown
        breakdown = "  ".join(f"{name}={val:.4f}" for name, val in terms.items())
        log.info("    └─ loss=%.4f │ %s", L, breakdown)

        return float(L)

    @staticmethod
    def _key(params: dict) -> tuple:
        return tuple(
            round(params[k], 8) if isinstance(params[k], float) else int(params[k])
            for k in sorted(params)
        )

    @staticmethod
    def _fmt(params: dict) -> str:
        return (f"T={params['T']:.2f} f={params['f']:.2f} "
                f"b={params['b']:.2f} N_lef={int(params['N_lef'])}")


# ---------------------------------------------------------------------------
# Lean CMA-ES  (Hansen 2016 tutorial notation)
# ---------------------------------------------------------------------------
class _CMAES:
    """
    Minimal pure-numpy CMA-ES.
    Operates in normalised [0,1]^n continuous space.
    Candidates are snapped to the nearest grid point before evaluation.
    """

    def __init__(self, x0: np.ndarray, sigma0: float, lam: int):
        n         = len(x0)
        self.n    = n
        self.lam  = lam
        self.mu   = lam // 2

        w_raw    = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w   = w_raw / w_raw.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()

        # step-size control constants
        self.cs    = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds    = (1 + 2 * max(0.0, math.sqrt((self.mueff-1)/(n+1)) - 1) + self.cs)
        self.chiN  = math.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        # covariance control constants
        self.cc   = (4 + self.mueff/n) / (n + 4 + 2*self.mueff/n)
        self.c1   = 2 / ((n + 1.3)**2 + self.mueff)
        self.cmu  = min(1 - self.c1,
                        2*(self.mueff - 2 + 1/self.mueff) / ((n+2)**2 + self.mueff))

        # state
        self.mean       = np.clip(x0.copy(), 0.0, 1.0)
        self.sigma      = sigma0
        self.C          = np.eye(n)
        self.pc         = np.zeros(n)
        self.ps         = np.zeros(n)
        self.invsqrtC   = np.eye(n)
        self._eigeneval = 0
        self._gen       = 0

    def ask(self) -> np.ndarray:
        """Sample lambda candidates in [0,1]^n."""
        try:
            L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.n))
        except np.linalg.LinAlgError:
            self.C = np.eye(self.n)
            L = np.eye(self.n)
        Z = np.random.randn(self.lam, self.n)
        return np.clip(self.mean + self.sigma * (Z @ L.T), 0.0, 1.0)

    def tell(self, candidates: np.ndarray, losses: np.ndarray):
        """Update CMA state given evaluated candidates and their losses."""
        n  = self.n
        mu = self.mu
        order  = np.argsort(losses)
        xbest  = candidates[order[:mu]]

        old_mean   = self.mean.copy()
        self.mean  = np.clip(self.w @ xbest, 0.0, 1.0)
        step       = (self.mean - old_mean) / (self.sigma + 1e-12)

        self.ps = ((1 - self.cs) * self.ps
                   + math.sqrt(self.cs*(2 - self.cs)*self.mueff)
                   * self.invsqrtC @ step)

        hs = (np.linalg.norm(self.ps)
              / math.sqrt(1 - (1 - self.cs)**(2*(self._gen+1)))
              / self.chiN) < (1.4 + 2/(n+1))

        self.pc = ((1 - self.cc) * self.pc
                   + hs * math.sqrt(self.cc*(2-self.cc)*self.mueff) * step)

        artmp  = (xbest - old_mean) / (self.sigma + 1e-12)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc)
                                + (1-hs)*self.cc*(2-self.cc)*self.C)
                  + self.cmu * (self.w * artmp.T) @ artmp)

        self.sigma *= math.exp((self.cs/self.ds)
                               * (np.linalg.norm(self.ps)/self.chiN - 1))
        self.sigma  = float(np.clip(self.sigma, 1e-4, 2.0))

        # eigendecomposition (amortised every ~n/c1 steps for efficiency)
        self._gen += 1
        thresh = self.lam / (10*(self.c1+self.cmu)*n + 1e-9)
        if self._gen - self._eigeneval > thresh:
            self._eigeneval = self._gen
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            vals, B = np.linalg.eigh(self.C)
            vals    = np.maximum(vals, 1e-8)
            self.invsqrtC = B @ np.diag(1.0/np.sqrt(vals)) @ B.T


# ---------------------------------------------------------------------------
# Stage 0 – Warm start
# ---------------------------------------------------------------------------
def _warm_start(evaluator: _Evaluator,
                grids: Dict[str, list],
                config: TunerConfig,
                rng: np.random.Generator) -> Tuple[dict, float, np.ndarray]:
    n_warm = config.n_warm
    log.info("Warm start: sampling %d random candidates to seed CMA-ES …", n_warm)
    log.info("  (each candidate runs a short %d-step simulation)",
             config.N_steps_short)

    pool      = [{k: rng.choice(grids[k]) for k in grids} for _ in range(n_warm)]
    losses    = []
    t_ws_start = time.time()
    for i, p in enumerate(_progress(pool, desc="Warm start", total=n_warm)):
        log.info("  ── Warm candidate %d/%d ──", i + 1, n_warm)
        l = evaluator(p)
        losses.append(l)
        elapsed = time.time() - t_ws_start
        eta_s   = elapsed / (i + 1) * (n_warm - i - 1)
        log.info("     loss=%.4f │ elapsed=%.1fmin │ ETA=%.1fmin",
                 l, elapsed / 60, eta_s / 60)
    losses = np.array(losses)
    order  = np.argsort(losses)

    best_p = pool[order[0]]
    best_l = losses[order[0]]

    log.info("")
    log.info("  Warm start ranking (best → worst):")
    for rank, idx in enumerate(order[:min(5, n_warm)]):
        marker = " ◀ BEST" if rank == 0 else ""
        log.info("    #%d  %s  loss=%.4f%s",
                 rank + 1, evaluator._fmt(pool[idx]), losses[idx], marker)
    if n_warm > 5:
        log.info("    … (%d more not shown)", n_warm - 5)

    # CMA initial mean = centroid of top-30% in unit space
    n_elite = max(1, int(0.3 * n_warm))
    elite_x = np.array([_to_unit(pool[i], grids) for i in order[:n_elite]])
    x0 = elite_x.mean(axis=0)
    x0_params = _from_unit(x0, grids)
    log.info("")
    log.info("  CMA-ES initial mean (centroid of top %d candidates): %s",
             n_elite, evaluator._fmt(x0_params))

    return best_p, best_l, x0


# ---------------------------------------------------------------------------
# Stage 1 – CMA-ES
# ---------------------------------------------------------------------------
def _run_cma(evaluator: _Evaluator,
             grids: Dict[str, list],
             config: TunerConfig,
             x0: np.ndarray,
             rng: np.random.Generator) -> Tuple[dict, float]:

    cma = _CMAES(x0, config.sigma0, config.lam)

    best_params = _from_unit(x0, grids)
    best_loss   = evaluator(best_params)

    ema_loss    = best_loss
    recent_ema: deque = deque(maxlen=config.patience + 1)

    # surrogate pre-screening kicks in after enough data has accumulated
    # (need at least 6 points for a reliable diagonal quadratic fit)
    USE_SURROGATE_AFTER_GEN = 3
    # reject candidates predicted worse than best + this margin
    SURROGATE_MARGIN = 0.08

    log.info("CMA-ES: max %d generations │ λ=%d candidates/gen │ σ₀=%.3f",
             config.max_generations, config.lam, config.sigma0)
    log.info("  Early stop : patience=%d gens │ min_delta=%.5f │ std_tol=%.5f",
             config.patience, config.min_delta, config.loss_std_tol)
    log.info("  Speed trick: surrogate pre-screening active after gen %d "
             "(reject margin=%.3f)", USE_SURROGATE_AFTER_GEN, SURROGATE_MARGIN)

    t_cma_start = time.time()

    for gen in _progress(range(config.max_generations),
                         desc="CMA-ES", total=config.max_generations):

        t_gen_start = time.time()
        use_surr    = gen >= USE_SURROGATE_AFTER_GEN

        log.info("")
        log.info("  ══ Generation %d/%d  │  σ=%.4f  │  surrogate=%s ══",
                 gen + 1, config.max_generations, cma.sigma,
                 "ON" if use_surr else "warming up")

        xs          = cma.ask()
        params_list = [_from_unit(x, grids) for x in xs]
        losses      = np.array([
            evaluator.evaluate(p, n_repeats=config.n_repeats,
                               use_surrogate=use_surr,
                               surrogate_reject_margin=SURROGATE_MARGIN)
            for p in params_list
        ])

        cma.tell(xs, losses)

        idx         = int(np.argmin(losses))
        gen_best_l  = losses[idx]
        gen_best_p  = params_list[idx]
        new_overall = gen_best_l < best_loss

        if new_overall:
            best_loss   = gen_best_l
            best_params = gen_best_p

        ema_loss = config.ema_alpha * gen_best_l + (1 - config.ema_alpha) * ema_loss
        recent_ema.append(ema_loss)

        t_gen = time.time() - t_gen_start
        t_tot = time.time() - t_cma_start

        # population summary
        log.info("  Population  min=%.4f  mean=%.4f  max=%.4f  std=%.4f",
                 float(losses.min()), float(losses.mean()),
                 float(losses.max()), float(losses.std()))

        # rank top-3 of this generation
        gen_order = np.argsort(losses)
        log.info("  Top candidates this generation:")
        for rank, gi in enumerate(gen_order[:min(3, len(params_list))]):
            tag = "  ◀ NEW OVERALL BEST ✓" if (rank == 0 and new_overall) else ""
            log.info("    #%d  %s  loss=%.4f%s",
                     rank + 1, evaluator._fmt(params_list[gi]), losses[gi], tag)

        log.info("  EMA=%.4f │ overall_best=%.4f  %s",
                 ema_loss, best_loss, evaluator._fmt(best_params))
        log.info("  Timing: gen=%.1fs │ CMA total=%.1fmin │ %s",
                 t_gen, t_tot / 60, evaluator.stats_str())

        # early stopping
        if len(recent_ema) == config.patience + 1:
            improvement = recent_ema[0] - recent_ema[-1]
            std_val     = float(np.std(recent_ema))
            if improvement < config.min_delta and std_val < config.loss_std_tol:
                log.info("")
                log.info("  ✋ Early stop: EMA_improvement=%.5f < %.5f  "
                         "AND  EMA_std=%.5f < %.5f  (gen %d)",
                         improvement, config.min_delta,
                         std_val, config.loss_std_tol, gen + 1)
                break

        if cma.sigma < 1e-3:
            log.info("  ✋ σ=%.2e collapsed — converged.", cma.sigma)
            break

    total_cma = time.time() - t_cma_start
    log.info("")
    log.info("  CMA-ES finished: %d generations │ %.1f min total", gen + 1, total_cma / 60)
    log.info("  Best found: loss=%.4f  %s", best_loss, evaluator._fmt(best_params))
    log.info("  Budget summary: %s", evaluator.stats_str())
    return best_params, best_loss


# ---------------------------------------------------------------------------
# Stage 2 – Neighbourhood polish
# ---------------------------------------------------------------------------
def _polish(evaluator: _Evaluator,
            grids: Dict[str, list],
            config: TunerConfig,
            params: dict,
            best_loss: float) -> Tuple[dict, float]:

    log.info("Polish: checking ±1 grid neighbours │ %d rounds │ "
             "%d repeats/neighbour (averaged to reduce noise)",
             config.n_polish_rounds, config.polish_n_repeats)
    log.info("  Starting point: %s  loss=%.4f",
             evaluator._fmt(params), best_loss)

    for rnd in range(config.n_polish_rounds):
        moved = False
        log.info("")
        log.info("  ── Polish round %d/%d  (current loss=%.4f) ──",
                 rnd + 1, config.n_polish_rounds, best_loss)
        log.info("  Current params: %s", evaluator._fmt(params))

        for key in sorted(grids.keys()):
            grid = grids[key]
            idx  = _grid_idx(params[key], grid)
            for di in (-1, +1):
                ni = idx + di
                if ni < 0 or ni >= len(grid):
                    continue
                cand      = copy.copy(params)
                cand[key] = grid[ni]
                log.info("    Testing %s: %.4g → %.4g …",
                         key, params[key], grid[ni])
                loss = evaluator.evaluate(cand, n_repeats=config.polish_n_repeats)
                delta = best_loss - loss
                if loss < best_loss - config.min_delta:
                    log.info("      ✓ Improved!  loss=%.4f  (Δ=%.5f)", loss, delta)
                    params    = cand
                    best_loss = loss
                    moved     = True
                else:
                    log.info("      ✗ No improvement  loss=%.4f  (Δ=%.5f)", loss, delta)

        if not moved:
            log.info("")
            log.info("  No neighbour improved in round %d — polish complete.", rnd + 1)
            break

    log.info("")
    log.info("  Polish result: %s  loss=%.4f",
             evaluator._fmt(params), best_loss)
    return params, best_loss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_parameter_search(sim, config: Optional[TunerConfig] = None) -> dict:
    """
    Run the full three-stage hyperparameter search for LoopSage.

    Parameters
    ----------
    sim : StochasticSimulation
        Fully preprocessed (J_loss, J, L, R, N_CTCF, h all set).
    config : TunerConfig, optional

    Returns
    -------
    best : dict
        Keys: T, f, b, N_lef, loss, and T_min if mode='Annealing'.
        kappa is fixed via TunerConfig.kappa — not in the dict.
    """
    if config is None:
        config = TunerConfig()
    if config.grids is None:
        config.grids = default_grids(sim.N_CTCF)

    grids = config.grids
    rng   = np.random.default_rng(42)

    W = 68
    log.info("=" * W)
    log.info("  LoopSage Parameter Tuner  —  warm-start → CMA-ES → polish")
    log.info("=" * W)
    log.info("  Simulation mode  : %s", config.mode)
    log.info("  Short-run budget : N_steps=%-6d  MC_step=%-4d  burnin=%d",
             config.N_steps_short, config.MC_step, config.burnin)
    log.info("  Fixed kappa      : %.4f  (not tuned)", config.kappa)
    if config.mode == "Annealing":
        log.info("  T_min fraction   : %.2f  (T_min = T × %.2f)",
                 config.T_min_fraction, config.T_min_fraction)
    log.info("")
    log.info("  Search grids:")
    for k, v in grids.items():
        log.info("    %-6s : %d values  [%.3g … %.3g]",
                 k, len(v), float(v[0]), float(v[-1]))
    log.info("")
    log.info("  Loss weights:")
    log.info("    w_dist    = %.2f  (distance-heatmap Pearson)", config.w_dist)
    log.info("    w_density = %.2f  (1D loop-density vs CTCF)", config.w_density)
    log.info("    w_lendist = %.2f  (loop-length KL divergence)", config.w_lendist)
    log.info("    w_unfold  = %.2f  (unfolding fraction)", config.w_unfold)
    log.info("    w_comp    = %.2f  (compartment agreement%s)",
             config.w_comp, "" if config.w_comp > 0 else " — disabled")
    log.info("")
    log.info("  CMA-ES settings:")
    log.info("    n_warm=%d  lam=%d  max_gen=%d  sigma0=%.3f",
             config.n_warm, config.lam, config.max_generations, config.sigma0)
    log.info("    patience=%d  min_delta=%.5f  loss_std_tol=%.5f",
             config.patience, config.min_delta, config.loss_std_tol)
    log.info("=" * W)

    ref            = _ReferenceData(sim, config)
    evaluator      = _Evaluator(sim, config, ref)
    t_search_start = time.time()

    # ── Stage 0 ───────────────────────────────────────────────────────
    log.info("")
    log.info("┌" + "─" * (W - 2) + "┐")
    log.info("│  STAGE 0 — Warm Start                                            │")
    log.info("└" + "─" * (W - 2) + "┘")
    best_p, best_l, x0 = _warm_start(evaluator, grids, config, rng)

    # ── Stage 1 ───────────────────────────────────────────────────────
    log.info("")
    log.info("┌" + "─" * (W - 2) + "┐")
    log.info("│  STAGE 1 — CMA-ES                                                │")
    log.info("└" + "─" * (W - 2) + "┘")
    log.info("  Starting from warm-start centroid.  Warm-start best: loss=%.4f", best_l)
    cma_p, cma_l = _run_cma(evaluator, grids, config, x0, rng)
    if cma_l < best_l:
        log.info("")
        log.info("  ✓ CMA-ES improved over warm start: %.4f → %.4f", best_l, cma_l)
        best_p, best_l = cma_p, cma_l
    else:
        log.info("")
        log.info("  Warm-start best (%.4f) beat CMA-ES best (%.4f) — keeping warm-start.",
                 best_l, cma_l)

    # ── Stage 2 ───────────────────────────────────────────────────────
    log.info("")
    log.info("┌" + "─" * (W - 2) + "┐")
    log.info("│  STAGE 2 — Neighbourhood Polish                                  │")
    log.info("└" + "─" * (W - 2) + "┘")
    best_p, best_l = _polish(evaluator, grids, config, best_p, best_l)

    T    = float(best_p["T"])
    best = {
        "T":     T,
        "f":     float(best_p["f"]),
        "b":     float(best_p["b"]),
        "N_lef": int(best_p["N_lef"]),
        "loss":  float(best_l),
    }
    if config.mode == "Annealing":
        best["T_min"] = T * config.T_min_fraction

    total_elapsed = time.time() - t_search_start
    log.info("")
    log.info("=" * W)
    log.info("  TUNING COMPLETE")
    log.info("=" * W)
    log.info("  Wall time       : %.1f min  (%.1f h)",
             total_elapsed / 60, total_elapsed / 3600)
    log.info("  Simulations run : %d  (cache hits: %d  surrogate skipped: %d)",
             evaluator.n_calls, evaluator.n_cache, evaluator.n_skipped)
    log.info("  Avg time/sim    : %.1f s",
             total_elapsed / max(evaluator.n_calls, 1))
    log.info("")
    log.info("  Best parameters found:")
    log.info("  %-8s = %.4f", "T", best["T"])
    if "T_min" in best:
        log.info("  %-8s = %.4f", "T_min", best["T_min"])
    log.info("  %-8s = %.4f", "f", best["f"])
    log.info("  %-8s = %.4f", "b", best["b"])
    log.info("  %-8s = %d",   "N_lef", best["N_lef"])
    log.info("  %-8s = %.4f  (fixed, not tuned)", "kappa", config.kappa)
    log.info("  %-8s = %.6f", "loss", best["loss"])
    log.info("=" * W)
    return best


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------
def best_params_to_kwargs(best: dict, kappa: float) -> dict:
    """
    Convert tuner output to kwargs for sim.run_energy_minimization().

    Example
    -------
    >>> best = run_parameter_search(sim, config)
    >>> sim.N_lef = best['N_lef']
    >>> sim.run_energy_minimization(
    ...     N_steps=200_000, MC_step=1_000, burnin=20_000,
    ...     mode=config.mode, viz=True, save=True,
    ...     **best_params_to_kwargs(best, kappa=config.kappa),
    ... )
    """
    kwargs = {"T": best["T"], "f": best["f"], "b": best["b"], "kappa": kappa}
    if "T_min" in best:
        kwargs["T_min"] = best["T_min"]
    return kwargs


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
def _demo():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from loopsage.stochastic_simulation import StochasticSimulation
    except ImportError:
        print("Could not import LoopSage.  Adjust sys.path.")
        return

    sim = StochasticSimulation(
        interaction_file="path/to/your.bedpe",
        chrom="chr6",
        region=[15_550_000, 16_850_000],
        out_dir="tuner_output",
        N_beads=200,
    )

    config = TunerConfig(
        mode="Annealing",
        kappa=1.0,
        N_steps_short=8_000,
        MC_step=400,
        burnin=2_000,
        n_warm=20,
        lam=10,
        max_generations=50,
        sigma0=0.3,
        n_repeats=1,
        patience=8,
        min_delta=5e-4,
        loss_std_tol=3e-3,
        ema_alpha=0.3,
        n_polish_rounds=4,
        polish_n_repeats=2,
        w_dist=1.0,
        w_density=0.5,
        w_lendist=0.3,
        w_unfold=0.2,
        w_comp=0.4,
        verbose=True,
    )

    best = run_parameter_search(sim, config)
    print("\n=== Best parameters ===")
    for k, v in best.items():
        print(f"  {k:8s} = {v}")

    sim.N_lef = best["N_lef"]
    sim.run_energy_minimization(
        N_steps=200_000, MC_step=1_000, burnin=20_000,
        mode=config.mode, viz=True, save=True,
        **best_params_to_kwargs(best, kappa=config.kappa),
    )


if __name__ == "__main__":
    _demo()