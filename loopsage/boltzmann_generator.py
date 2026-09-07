"""
boltzmann_generator.py

A Boltzmann Generator (normalizing-flow-based sampler) for the LoopSage
Hamiltonian, restricted to three physical terms:

    E(m, n) = f_norm * sum_i g(n_i - m_i)                          [folding]
            + k_norm * sum_{i<j} K_soft(m_i, n_i, m_j, n_j)        [crossing]
            + b_norm * sum_i [ L(m_i) + R(n_i) ]                   [CTCF binding]

Anchors (m_i, n_i) are reached through a bounded sigmoid reparametrization
of the flow's raw output, so bead centers and log loop-lengths are always
finite and in-range by construction -- no clamping, no possibility of an
inf/nan reaching a lookup index.

Two classes:

    LoopSage_Generator
        A RealNVP normalizing flow trained by reverse-KL against the
        differentiable LoopSage energy. Produces approximate i.i.d.
        Boltzmann samples of LEF configurations (m_i, n_i) in one pass.

    NeuralMCMCSampler
        Wraps a trained LoopSage_Generator as the proposal distribution in
        a Metropolis-Hastings chain, using the flow's tractable density for
        the Hastings correction -- exact w.r.t. the true Boltzmann
        distribution, with large non-local proposal jumps.
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .logger import get_logger
from .preproc import binding_vectors_from_bedpe

log = get_logger(__name__)


# ============================================================================
# 0. Device handling
# ============================================================================

def resolve_device(device: str = "cpu") -> torch.device:
    """
    Resolve a user-requested device string ("cpu", "cuda", "cuda:0", ...)
    into a torch.device, falling back to CPU with a warning if CUDA was
    requested but is not actually available. CPU is the default so the
    generator runs everywhere out of the box; pass device="cuda" explicitly
    to use a GPU.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning(f"Requested device='{device}' but CUDA is not available -> falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


# ============================================================================
# 1. Differentiable LoopSage energy (folding + crossing + CTCF binding only)
# ============================================================================

@dataclass
class LoopSageEnergyConfig:
    """
    Physical / numerical hyperparameters of the LoopSage Hamiltonian, on the
    same scale as StochasticSimulation.run_energy_minimization().

    N_beads, N_lef: polymer resolution and number of LEFs.
    f, b, kappa: folding / binding / crossing coefficients -- pure relative
        weights. fold_norm/bind_norm/k_norm are all pinned to the same
        N_beads energy scale (see each method), so at f=b=kappa=1 no term
        structurally dominates the others; raising one coefficient above 1
        is what actually shifts the balance.
    crossing_softness: width (beads) of the smooth crossing-indicator
        surrogate; smaller = closer to the true discrete rule.
    crossing_safety: how many multiples of the whole system's fold/bind
        energy scale (N_beads) a single fully-crossing pair costs, before
        applying kappa. Crossings are meant to be a near-hard constraint,
        so this is deliberately >> 1 (default 10).
    T: Boltzmann temperature for reverse-KL training.
    min_loop_length, max_loop_length: hard bounds on loop length, enforced
        by construction via the flow's bounded reparametrization.
    rectify_binding: preproc's contrastive L/R normalization can leave
        background (no-CTCF) beads slightly negative instead of 0, which
        flips into a spurious positive E_bind. Clips L/R at 0 so E_bind
        stays <= 0 everywhere. Set False to match stochastic_simulation.py
        exactly.
    fold_mode: how loop length enters the folding term, per LEF:
        "log" (default, matches stochastic_simulation.py): fold_norm*log(ell).
            Diminishing marginal reward for growth -- most valuable while a
            loop is still short.
        "linear": fold_norm*ell. Constant marginal reward regardless of
            current length.
        "quadratic": fold_norm*ell**2. Growing reward -- once a loop starts
            growing it's pushed increasingly hard to keep growing.
    """
    N_beads: int
    N_lef: int
    f: float = 1.0
    b: float = 1.0
    kappa: float = 1.0
    crossing_softness: float = 2.0
    crossing_safety: float = 10.0
    T: float = 1.0
    min_loop_length: float = 2.0
    max_loop_length: Optional[float] = None
    rectify_binding: bool = True
    fold_mode: str = "log"

    def __post_init__(self):
        if self.max_loop_length is None:
            self.max_loop_length = float(self.N_beads - 1)
        if not (0 < self.min_loop_length < self.max_loop_length):
            raise ValueError(
                f"Require 0 < min_loop_length ({self.min_loop_length}) < "
                f"max_loop_length ({self.max_loop_length})."
            )
        if self.fold_mode not in ("log", "linear", "quadratic"):
            raise ValueError(f"fold_mode must be 'log', 'linear' or 'quadratic', got {self.fold_mode!r}")

    def fold_norm(self) -> float:
        """E_fold = fold_norm * sum_i log(ell_i). At the "typical" evenly-
        tiled configuration (ell ~ N_beads/N_lef), this gives
        E_fold = -N_beads*f -- the common reference scale bind_norm and
        k_norm are also pinned to, so f/b/kappa trade off directly."""
        return -self.N_beads * self.f / (self.N_lef * np.log(self.N_beads / self.N_lef))

    def bind_norm(self) -> float:
        """E_bind = bind_norm * sum_i [L(m_i)+R(n_i)], L,R roughly in [0,1].
        Normalized by N_lef -- the number of terms actually summed here --
        not by the CTCF-peak count: at full occupancy (every LEF sitting on
        a maximal motif on both sides), this gives E_bind = -N_beads*b, the
        same reference scale as E_fold. Dividing by the peak count instead
        (as an earlier version did) let a region with many CTCF peaks but
        few modeled LEFs silently shrink E_bind toward 0 regardless of b --
        a data-dependent imbalance with nothing to do with the physics."""
        return -self.N_beads * self.b / (2 * self.N_lef)

    def k_norm(self) -> float:
        """E_cross = k_norm * sum_{i<j} K_soft(...), K_soft in [0,1] per
        pair. Pinned to the same N_beads reference (instead of a fixed
        constant like 1e4, which silently weakens relative to fold/bind as
        N_beads grows): a single fully-crossing pair costs crossing_safety
        times the whole system's fold/bind budget at kappa=1 -- a strong,
        near-hard constraint at any problem size, not just the one it was
        tuned on."""
        return self.kappa * self.N_beads * self.crossing_safety


def generate_initial_configuration(N_beads: int, N_lef: int, mode: str = "random",
                                    min_loop_length: float = 2.0, batch_size: int = 1,
                                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline (non-learned) loop configurations -- for sanity-checking the
    energy function, or as a fixed reference point to compare a trained
    ensemble against. Three modes:
      "parallel": N_lef equal, evenly-spaced, side-by-side loops -- no
                  crossings, no nesting, every loop an orphan.
      "random":   independent uniform random (m, n) pairs -- may cross,
                  nest, or overlap in any way.
      "minimal":  every loop at min_loop_length, at random positions.
    Returns (ms, ns): (batch_size, N_lef) int arrays.
    """
    rng = np.random.default_rng(seed)
    min_len = max(int(min_loop_length), 1)
    ms = np.zeros((batch_size, N_lef), dtype=np.int64)
    ns = np.zeros((batch_size, N_lef), dtype=np.int64)

    for b in range(batch_size):
        if mode == "parallel":
            span = N_beads // N_lef
            length = max(span - 1, min_len)
            starts = np.arange(N_lef) * span
            ms[b] = starts
            ns[b] = np.minimum(starts + length, N_beads - 1)
        elif mode == "minimal":
            starts = rng.choice(N_beads - min_len, size=N_lef, replace=False)
            ms[b] = starts
            ns[b] = starts + min_len
        elif mode == "random":
            m = rng.integers(0, N_beads - min_len - 1, size=N_lef)
            n = rng.integers(m + min_len, N_beads, size=N_lef)
            ms[b], ns[b] = m, n
        else:
            raise ValueError(f"mode must be 'parallel', 'random' or 'minimal', got {mode!r}")
    return ms, ns


class LoopSageEnergy(nn.Module):
    """
    Differentiable re-implementation of the LoopSage Hamiltonian, using a
    bounded reparametrization of (m_i, n_i) so raw flow output can never
    blow up into inf/nan (the fix for the earlier CUDA index crash):

        c = sigmoid(c_raw) * (N_beads-1)     -- loop center, in-range
        u = log(min_len) + sigmoid(u_raw)*(log(max_len)-log(min_len))
        ell = exp(u)                          -- loop length, in-range
        m = c - ell/2, n = c + ell/2          -- anchors (then clamped)

    L, R : (N_beads,) binding tracks from preproc.py, interpolated to
           continuous bead coordinates.
    """

    def __init__(self, L: np.ndarray, R: np.ndarray, cfg: LoopSageEnergyConfig):
        super().__init__()
        assert L.shape == R.shape, "L and R must have the same shape"
        self.cfg = cfg
        self.N_beads = cfg.N_beads
        self.N_lef = cfg.N_lef

        if cfg.rectify_binding:
            # See LoopSageEnergyConfig.rectify_binding: clip contrastive-
            # normalized L/R at 0 so "no CTCF here" reads as exactly 0
            # rather than a stray negative value that would otherwise flip,
            # through the negative bind_norm, into a spurious *positive*
            # (repulsive) E_bind at ordinary background beads.
            n_clipped = int(np.sum(L < 0) + np.sum(R < 0))
            L = np.clip(L, 0.0, None)
            R = np.clip(R, 0.0, None)
            log.info(f"rectify_binding=True: clipped {n_clipped} negative L/R entries to 0 "
                     f"(out of {2 * len(L)}) -> E_bind is now <= 0 everywhere by construction.")

        self.register_buffer("L", torch.as_tensor(L, dtype=torch.float32))
        self.register_buffer("R", torch.as_tensor(R, dtype=torch.float32))

        self.fold_norm = cfg.fold_norm()
        self.bind_norm = cfg.bind_norm()
        self.softness = cfg.crossing_softness
        self.fold_mode = cfg.fold_mode

        # Single mutable crossing weight. Starts at the full cfg value;
        # train_flow may overwrite it directly during warm-up (ramping from
        # 0 up to this same target, kept in _k_norm_target) so folding/
        # binding can shape the distribution before the much larger
        # crossing term dominates -- there's only ever one number in play
        # (k_norm), just updated over time.
        self._k_norm_target = cfg.k_norm()
        self.k_norm = self._k_norm_target

        # bounded-domain constants (fixed, not learned)
        self.c_min, self.c_max = 0.0, float(self.N_beads - 1)
        self.u_min = float(math.log(cfg.min_loop_length))
        self.u_max = float(math.log(cfg.max_loop_length))
        self.eps = 1e-6

        # Strict-upper-triangular pair mask (i<j), precomputed once: N_lef is
        # fixed for the life of this module, so crossing_energy/crossing_count
        # (called every training step) reuse this instead of reallocating an
        # (N,N) boolean tensor -- and it already excludes the diagonal, so no
        # separate eye-mask is needed.
        self.register_buffer(
            "_pair_mask", torch.triu(torch.ones(self.N_lef, self.N_lef, dtype=torch.bool), diagonal=1)
        )

    # ---- bounded reparametrization: raw flow output -> physical (c, u) ----
    def squash(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """raw (B, 2*N_lef) -> bounded (c, u), each (B, N_lef).

        Defensive nan_to_num guards against a stray nan/inf that could in
        principle reach here from an unstable optimizer step upstream (the
        squash itself cannot produce one from any finite or infinite input,
        since sigmoid saturates rather than overflowing)."""
        x_raw = torch.nan_to_num(x_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        c_raw, u_raw = x_raw[:, : self.N_lef], x_raw[:, self.N_lef:]
        c = self.c_min + torch.sigmoid(c_raw) * (self.c_max - self.c_min)
        u = self.u_min + torch.sigmoid(u_raw) * (self.u_max - self.u_min)
        return c, u

    def cn_to_mn(self, c: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """bounded center/log-length -> (m, n), both (..., N_lef)."""
        ell = torch.exp(u)
        m = (c - 0.5 * ell).clamp(self.c_min, self.c_max)
        n = (c + 0.5 * ell).clamp(self.c_min, self.c_max)
        n = torch.maximum(n, m + self.eps)  # guard the rare edge-clamp case
        return m, n

    def decode(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """raw flow output -> (c, u, m, n), all guaranteed finite and in-bounds."""
        c, u = self.squash(x_raw)
        m, n = self.cn_to_mn(c, u)
        return c, u, m, n

    # ---- differentiable linear interpolation of L/R at continuous coords ----
    def _interp(self, table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=self.N_beads - 1.0, neginf=0.0)
        x = x.clamp(0.0, self.N_beads - 1.0 - 1e-4)
        x0 = torch.floor(x)
        frac = x - x0
        x0 = x0.long()
        x1 = (x0 + 1).clamp(max=self.N_beads - 1)
        v0 = table[x0]
        v1 = table[x1]
        return v0 * (1 - frac) + v1 * frac

    def folding_energy(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """E_fold = fold_norm * sum_i g(ell_i), g set by fold_mode:
        "log" (default) = log(ell) -- diminishing reward for growth;
        "linear" = ell -- constant reward per bead of growth;
        "quadratic" = ell**2 -- reward grows the longer a loop already is."""
        ell = (n - m).clamp(min=self.eps)
        if self.fold_mode == "log":
            g = torch.log(ell)
        elif self.fold_mode == "linear":
            g = ell
        else:  # "quadratic"
            g = ell ** 2
        return self.fold_norm * g.sum(dim=-1)

    def binding_energy(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """E_bind = bind_norm * sum_i [ L(m_i) + R(n_i) ], bind_norm < 0.
        With rectify_binding (default), L/R >= 0 everywhere (0 away from
        CTCF motifs, positive at real sites), so this is <= 0 for every
        configuration and only dips negative where an anchor actually sits
        on a motif -- never positive."""
        Lm = self._interp(self.L, m)
        Rn = self._interp(self.R, n)
        return self.bind_norm * (Lm + Rn).sum(dim=-1)

    def crossing_energy(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Smooth surrogate for a strict crossing: mi<mj<ni<nj or its
        mirror. Nesting (mi<=mj<=nj<=ni, a parent with any number of
        children) never satisfies this -- it needs ni<nj, which containment
        contradicts -- so children are excluded by the definition itself,
        with no extra gating required. (An earlier version added a soft
        "is this pair nested" gate to also suppress the residual softness
        bleed right at tight boundaries; that gate could also partially
        discount genuine near-boundary crossings, weakening the very
        penalty meant to drive crossings to 0, so it was removed -- any
        leftover bleed here is just the ordinary smoothness of a sigmoid
        relaxation, not a nesting penalty.) Always >= 0; scaled by k_norm
        (see the k_norm comment in __init__ for the warm-up ramp)."""
        s = self.softness

        mi, ni = m.unsqueeze(2), n.unsqueeze(2)
        mj, nj = m.unsqueeze(1), n.unsqueeze(1)

        def sig(x):
            return torch.sigmoid(x / s)

        lt = lambda a, b: sig(b - a)

        cross1 = lt(mi, mj) * lt(mj, ni) * lt(ni, nj)
        cross2 = lt(mj, mi) * lt(mi, nj) * lt(nj, ni)
        pair_penalty = (cross1 + cross2) * self._pair_mask

        crossing = pair_penalty.sum(dim=(1, 2))
        return self.k_norm * crossing

    @torch.no_grad()
    def crossing_count(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        Hard crossing count: pair (i,j) counts iff mi<mj<ni<nj (or the
        mirror mj<mi<nj<ni) -- a genuine partial overlap. Nesting
        (one loop fully inside another) is NOT a crossing. Diagnostic
        only; matches the strict-overlap definition used in
        crossing_energy(). Returns (B,) counts.
        """
        m_i = torch.round(m).long()
        n_i = torch.round(n).long()

        mi, ni = m_i.unsqueeze(2), n_i.unsqueeze(2)
        mj, nj = m_i.unsqueeze(1), n_i.unsqueeze(1)

        cross1 = (mi < mj) & (mj < ni) & (ni < nj)
        cross2 = (mj < mi) & (mi < nj) & (nj < ni)
        k = (cross1 | cross2) & self._pair_mask
        return k.sum(dim=(1, 2)).float()

    def energy_from_mn(self, m: torch.Tensor, n: torch.Tensor) -> dict:
        """Same breakdown as energy_components(), but starting directly
        from (m, n) pairs instead of raw flow output -- for evaluating
        baseline/synthetic configurations (see generate_initial_configuration),
        not just ones sampled from the flow."""
        E_fold = self.folding_energy(m, n)
        E_cross = self.crossing_energy(m, n)
        E_bind = self.binding_energy(m, n)
        n_crossings = self.crossing_count(m, n)
        return {
            "m": m, "n": n,
            "E_fold": E_fold, "E_cross": E_cross, "E_bind": E_bind,
            "E_total": E_fold + E_cross + E_bind,
            "n_crossings": n_crossings,
        }

    def energy_components(self, x_raw: torch.Tensor) -> dict:
        """Full breakdown, useful for diagnostics/plots. x_raw: (B, 2*N_lef)."""
        c, u, m, n = self.decode(x_raw)
        E_fold = self.folding_energy(m, n)
        E_cross = self.crossing_energy(m, n)
        E_bind = self.binding_energy(m, n)
        n_crossings = self.crossing_count(m, n)
        return {
            "m": m, "n": n, "c": c, "u": u,
            "E_fold": E_fold, "E_cross": E_cross, "E_bind": E_bind,
            "E_total": E_fold + E_cross + E_bind,
            "n_crossings": n_crossings,
        }

    def energy(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Total differentiable LoopSage energy. x_raw: (B, 2*N_lef) raw flow
        output -> returns (B,) energy per configuration."""
        c, u, m, n = self.decode(x_raw)
        return (self.folding_energy(m, n) + self.crossing_energy(m, n)
                + self.binding_energy(m, n))


# ============================================================================
# 2. RealNVP-style coupling-layer normalizing flow
# ============================================================================

class _ConditionerMLP(nn.Module):
    """Small MLP producing (log_scale, shift) for an affine coupling layer."""

    def __init__(self, dim_in: int, dim_out: int, hidden: int = 128, n_hidden_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(dim_in, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, dim_out)]
        self.net = nn.Sequential(*layers)
        # zero-init last layer -> flow starts close to identity (stabler training)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class _EquivariantConditioner(nn.Module):
    """
    Permutation-equivariant conditioner over a set of N_lef scalars (one
    per LEF). Each output depends only on its own input and on
    permutation-invariant summaries (mean, max) of all the others -- so
    relabeling which LEF is #3 vs #17 relabels the output the same way.
    A plain dense MLP (_ConditionerMLP) does NOT have this property: its
    weights privilege specific input positions, silently breaking the
    physical fact that LEFs are interchangeable and biasing which ones
    the flow finds it easy to grow vs. leave stuck.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Linear(3 * hidden, 2)
        nn.init.zeros_(self.rho.weight)
        nn.init.zeros_(self.rho.bias)

    def forward(self, x_cond: torch.Tensor) -> torch.Tensor:
        B, N = x_cond.shape
        h = self.phi(x_cond.unsqueeze(-1))                          # (B, N, hidden)
        pooled_mean = h.mean(dim=1, keepdim=True).expand(-1, N, -1)
        pooled_max = h.amax(dim=1, keepdim=True).expand(-1, N, -1)
        st = self.rho(torch.cat([h, pooled_mean, pooled_max], dim=-1))  # (B, N, 2)
        log_s, t = st[..., 0], st[..., 1]
        return torch.cat([log_s, t], dim=-1)  # (B, 2N), same layout _ConditionerMLP returns


class AffineCouplingLayer(nn.Module):
    """
    RealNVP affine coupling layer with a binary mask.

        x_a = x[mask], x_b = x[~mask]
        y_a = x_a
        y_b = x_b * exp(log_s(x_a)) + t(x_a)

    log_s is squashed with `max_log_scale * tanh(...)` so the per-layer
    scale factor is always bounded in [exp(-max_log_scale), exp(max_log_scale)];
    the shift t is left unbounded (as in a standard RealNVP) since it is the
    *downstream* sigmoid squashing in LoopSageEnergy -- not this layer --
    that is responsible for keeping decoded bead coordinates finite.
    """

    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 128, max_log_scale: float = 2.0,
                 equivariant: bool = True):
        super().__init__()
        self.register_buffer("mask", mask.float())
        n_transformed = int((1 - mask).sum().item())
        n_condition = int(mask.sum().item())
        if equivariant and n_condition == n_transformed:
            self.conditioner = _EquivariantConditioner(hidden=hidden)
        else:
            self.conditioner = _ConditionerMLP(n_condition, 2 * n_transformed, hidden=hidden)
        self.dim = dim
        self.max_log_scale = max_log_scale
        self.transform_idx = torch.where(mask == 0)[0]
        self.condition_idx = torch.where(mask == 1)[0]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cond = x[:, self.condition_idx]
        st = self.conditioner(x_cond)
        log_s, t = st.chunk(2, dim=-1)
        log_s = self.max_log_scale * torch.tanh(log_s)

        y = x.clone()
        y[:, self.transform_idx] = x[:, self.transform_idx] * torch.exp(log_s) + t
        log_det = log_s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_cond = y[:, self.condition_idx]
        st = self.conditioner(y_cond)
        log_s, t = st.chunk(2, dim=-1)
        log_s = self.max_log_scale * torch.tanh(log_s)

        x = y.clone()
        x[:, self.transform_idx] = (y[:, self.transform_idx] - t) * torch.exp(-log_s)
        log_det = -log_s.sum(dim=-1)
        return x, log_det


class RealNVPFlow(nn.Module):
    """A stack of alternating-mask affine coupling layers mapping a standard
    Gaussian base density to the flow's raw output density. Dimensionality
    is 2*N_lef (center + log-length per LEF, in pre-squash coordinates).

    The mask always splits [c-block | u-block] (each N_lef long), so with
    equivariant=True (default) every layer's conditioner is the
    permutation-equivariant one -- the whole flow then treats LEFs as an
    interchangeable set, matching the physical symmetry of the energy,
    instead of privileging specific LEF indices."""

    def __init__(self, dim: int, n_layers: int = 8, hidden: int = 128, max_log_scale: float = 2.0,
                 equivariant: bool = True):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[: dim // 2] = 1
            else:
                mask[dim // 2:] = 1
            layers.append(AffineCouplingLayer(dim, mask, hidden=hidden, max_log_scale=max_log_scale,
                                               equivariant=equivariant))
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z (base) -> x (raw flow space). Returns (x, log_det_total)."""
        x = z
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            x, log_det = layer.forward(x)
            log_det_total = log_det_total + log_det
        return x, log_det_total

    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x (raw flow space) -> z (base). Returns (z, log_det_total)."""
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det_inv = self.inverse(x)
        base_log_prob = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        return base_log_prob + log_det_inv

    def sample(self, n: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.dim, device=device)
        x, log_det_fwd = self.forward(z)
        base_log_prob = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        log_prob_x = base_log_prob - log_det_fwd
        return x, log_prob_x


# ============================================================================
# 3. LoopSage_Generator — the Boltzmann Generator itself
# ============================================================================

class LoopSage_Generator(nn.Module):
    """
    Boltzmann Generator for the LoopSage Hamiltonian (folding + crossing +
    CTCF binding only -- no data/Ising terms).

    Parameters
    ----------
    L, R : np.ndarray (N_beads,)
        CTCF binding potentials, as produced by preproc.binding_vectors_from_bedpe
        (or _bed / _narrowpeak).
    cfg : LoopSageEnergyConfig
    n_layers, hidden, max_log_scale : flow architecture size / stability knobs.
    device : str
        "cpu" (default) or "cuda"/"cuda:0" etc. Resolved via resolve_device(),
        which falls back to CPU with a warning if CUDA was requested but is
        unavailable.
    seed : Optional[int]
        If set, seeds torch's RNG for reproducible training/sampling.
    """

    def __init__(self, L: np.ndarray, R: np.ndarray, cfg: LoopSageEnergyConfig,
                 n_layers: int = 8, hidden: int = 128, max_log_scale: float = 2.0,
                 device: str = "cpu", seed: Optional[int] = None,
                 equivariant_flow: bool = True):
        super().__init__()
        self.cfg = cfg
        self.device = resolve_device(device)
        if seed is not None:
            torch.manual_seed(seed)

        self.energy_fn = LoopSageEnergy(L, R, cfg).to(self.device)
        self.flow = RealNVPFlow(dim=2 * cfg.N_lef, n_layers=n_layers, hidden=hidden,
                                 max_log_scale=max_log_scale, equivariant=equivariant_flow).to(self.device)
        self._arch_kwargs = dict(n_layers=n_layers, hidden=hidden, max_log_scale=max_log_scale,
                                  equivariant_flow=equivariant_flow)

        self.history = {
            "kl_loss": [], "mean_energy": [],
            "E_fold": [], "E_cross": [], "E_bind": [], "n_crossings": [],
            "batch_std": [], "T_eff": [], "length_cv": [], "skipped_steps": [],
        }

    # ---------------- core training objective ----------------
    def reverse_kl_loss(self, batch_size: int, T: Optional[float] = None,
                         lambda_diversity: float = 0.0) -> Tuple[torch.Tensor, dict]:
        """
        Reverse-KL loss: E_x~q[ E(x)/T + log q(x) ], using the flow's own
        samples so it's differentiable end-to-end.

        T: temperature for this call (defaults to cfg.T) -- lets train_flow
        anneal a hotter training temperature early on.

        lambda_diversity: weight of a batch-diversity penalty on anchor
        variance, to counter mode collapse (reverse-KL only needs to cover
        SOME modes of the target, not all of them). 0 disables it.

        Returns (loss, metrics) where metrics has the energy breakdown and
        batch_std for logging.
        """
        T = self.cfg.T if T is None else T
        z = torch.randn(batch_size, self.flow.dim, device=self.device)
        x, log_det_fwd = self.flow.forward(z)
        base_log_prob = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        log_prob_x = base_log_prob - log_det_fwd

        comp = self.energy_fn.energy_components(x)
        E = comp["E_total"]
        loss = (E / T + log_prob_x).mean()

        m, n = comp["m"], comp["n"]
        batch_std = 0.5 * (m.std(dim=0) + n.std(dim=0)).mean()  # mean over LEFs of anchor std across the batch
        if lambda_diversity > 0:
            diversity_loss = -(torch.log(m.var(dim=0) + 1e-3) + torch.log(n.var(dim=0) + 1e-3)).mean()
            loss = loss + lambda_diversity * diversity_loss

        # Diagnostic only: how much loop length varies within one sample.
        # Near 0 means every LEF has (nearly) the same length -- a sign
        # loops aren't extruding freely.
        with torch.no_grad():
            length = n - m
            length_cv = (length.std(dim=1) / (length.mean(dim=1) + 1e-6)).mean()

        metrics = {
            "mean_E": E.mean().item(),
            "E_fold": comp["E_fold"].mean().item(),
            "E_cross": comp["E_cross"].mean().item(),
            "E_bind": comp["E_bind"].mean().item(),
            "n_crossings": comp["n_crossings"].mean().item(),
            "batch_std": batch_std.item(),
            "length_cv": length_cv.item(),
        }
        return loss, metrics

    # ---------------- training loop ----------------
    def train_flow(self, n_steps: int = 5000, batch_size: int = 256,
                   lr: float = 1e-3, log_every: int = 200,
                   grad_clip: float = 5.0, max_consecutive_nonfinite: int = 20,
                   kappa_warmup_steps: int = 500,
                   temp_anneal_factor: float = 1.0, temp_anneal_steps: int = 0,
                   lambda_diversity: float = 0.0):
        """
        Reverse-KL training loop with a live progress bar (loss + energy
        breakdown) and a periodic log line every `log_every` steps.

        kappa_warmup_steps: ramps the crossing penalty from 0 to full
        strength, so folding/binding can shape the distribution before the
        much larger crossing term dominates. 0 disables it.

        temp_anneal_factor/steps + lambda_diversity: both fight mode
        collapse (reverse-KL only needs to cover some modes of this
        multimodal energy, so training tends to collapse onto one LEF
        arrangement regardless of sampling T). Annealing trains "hot" first
        then cools to cfg.T; lambda_diversity directly penalizes low anchor
        variance within a batch. Watch history["batch_std"] -- it should
        stay well above 0, not decay to 0.

        Non-finite losses are skipped (logged) rather than crashing; after
        `max_consecutive_nonfinite` in a row, training aborts.
        """
        opt = torch.optim.Adam(self.flow.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

        log.info(f"Training LoopSage_Generator flow for {n_steps} steps "
                 f"(batch_size={batch_size}, lr={lr}, device={self.device}, "
                 f"kappa_warmup_steps={kappa_warmup_steps}, "
                 f"temp_anneal_factor={temp_anneal_factor}, temp_anneal_steps={temp_anneal_steps}, "
                 f"lambda_diversity={lambda_diversity})")

        t0 = time.time()
        consecutive_nonfinite = 0

        pbar = tqdm(range(1, n_steps + 1), desc="Training BG", unit="step",
                    dynamic_ncols=True, leave=True)
        for step in pbar:
            opt.zero_grad()

            ramp = min(1.0, step / kappa_warmup_steps) if kappa_warmup_steps > 0 else 1.0
            self.energy_fn.k_norm = self.energy_fn._k_norm_target * ramp

            if temp_anneal_steps > 0 and temp_anneal_factor > 1.0:
                frac = max(0.0, 1.0 - step / temp_anneal_steps)
                T_eff = self.cfg.T * (1.0 + (temp_anneal_factor - 1.0) * frac)
            else:
                T_eff = self.cfg.T

            loss, metrics = self.reverse_kl_loss(batch_size, T=T_eff, lambda_diversity=lambda_diversity)
            metrics["T_eff"] = T_eff

            if not torch.isfinite(loss):
                consecutive_nonfinite += 1
                tqdm.write(f"[step {step}] non-finite loss ({loss.item()}) -> skipping "
                           f"optimizer step ({consecutive_nonfinite}/{max_consecutive_nonfinite}).")
                self.history["skipped_steps"].append(step)
                if consecutive_nonfinite >= max_consecutive_nonfinite:
                    raise RuntimeError(
                        f"Training aborted: {consecutive_nonfinite} consecutive non-finite losses. "
                        "Try a smaller learning rate, a smaller max_log_scale, or gradient clipping."
                    )
                continue
            consecutive_nonfinite = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), grad_clip)
            opt.step()
            scheduler.step()

            self.history["kl_loss"].append(loss.item())
            self.history["mean_energy"].append(metrics["mean_E"])
            self.history["E_fold"].append(metrics["E_fold"])
            self.history["E_cross"].append(metrics["E_cross"])
            self.history["E_bind"].append(metrics["E_bind"])
            self.history["n_crossings"].append(metrics["n_crossings"])
            self.history["batch_std"].append(metrics["batch_std"])
            self.history["T_eff"].append(metrics["T_eff"])
            self.history["length_cv"].append(metrics["length_cv"])

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "<E>": f"{metrics['mean_E']:.3f}",
                "E_fold": f"{metrics['E_fold']:.3f}",
                "E_bind": f"{metrics['E_bind']:.3f}",
                "E_cross": f"{metrics['E_cross']:.3f}",
                "#cross": f"{metrics['n_crossings']:.2f}",
                "std": f"{metrics['batch_std']:.2f}",
                "T_eff": f"{metrics['T_eff']:.2f}",
                "len_cv": f"{metrics['length_cv']:.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}",
            })

            if step % log_every == 0 or step == 1:
                elapsed = time.time() - t0
                tqdm.write(
                    f"[step {step:5d}/{n_steps}] loss={loss.item():.4f}  "
                    f"<E>={metrics['mean_E']:.4f}  E_fold={metrics['E_fold']:.4f}  "
                    f"E_bind={metrics['E_bind']:.4f}  E_cross={metrics['E_cross']:.4f}  "
                    f"#crossings={metrics['n_crossings']:.2f}  batch_std={metrics['batch_std']:.2f}  "
                    f"T_eff={metrics['T_eff']:.2f}  length_cv={metrics['length_cv']:.2f}  "
                    f"({elapsed:.1f}s elapsed)"
                )

        n_skipped = len(self.history["skipped_steps"])
        if n_skipped:
            log.warning(f"Training complete with {n_skipped} skipped non-finite step(s).")
        else:
            log.info("Training complete (no non-finite steps).")

    # ---------------- sampling & discretization ----------------
    def sample_continuous(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw flow samples (pre-squash). Returns (x_raw, log_prob_x)."""
        self.flow.eval()
        with torch.no_grad():
            x, log_prob = self.flow.sample(n, device=self.device)
        self.flow.train()
        return x, log_prob

    def cu_to_mn_discrete(self, x_raw: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw flow samples to integer bead positions (m, n). Thanks to
        the bounded squash() applied inside energy_fn.decode(), m and n are
        already guaranteed finite and within [0, N_beads-1] before rounding
        -- this is a plain discretization step, not a repair step.
        Non-crossing is NOT hard-enforced here; use NeuralMCMCSampler for an
        exact, crossing-aware chain.
        """
        with torch.no_grad():
            _, _, m, n = self.energy_fn.decode(x_raw)
        m_int = torch.round(m).long().clamp(0, self.cfg.N_beads - 1)
        n_int = torch.round(n).long().clamp(0, self.cfg.N_beads - 1)
        bad = n_int <= m_int
        n_int = torch.where(bad, (m_int + 1).clamp(max=self.cfg.N_beads - 1), n_int)
        return m_int.cpu().numpy(), n_int.cpu().numpy()

    # ---------------- persistence ----------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "flow_state_dict": self.flow.state_dict(),
            "cfg": self.cfg.__dict__,
            "arch_kwargs": self._arch_kwargs,
            "L": self.energy_fn.L.cpu().numpy(),
            "R": self.energy_fn.R.cpu().numpy(),
            "history": self.history,
        }, path)
        log.info(f"Saved LoopSage_Generator -> {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LoopSage_Generator":
        dev = resolve_device(device)
        ckpt = torch.load(path, map_location=dev)
        cfg = LoopSageEnergyConfig(**ckpt["cfg"])
        arch_kwargs = ckpt.get("arch_kwargs", {})
        model = cls(ckpt["L"], ckpt["R"], cfg, device=device, **arch_kwargs)
        model.flow.load_state_dict(ckpt["flow_state_dict"])
        model.history = ckpt.get("history", model.history)
        log.info(f"Loaded LoopSage_Generator <- {path}")
        return model


# ============================================================================
# 4. NeuralMCMCSampler — flow-proposed Metropolis-Hastings
# ============================================================================

class NeuralMCMCSampler:
    """
    Uses a trained LoopSage_Generator as a *global* proposal distribution
    inside a Metropolis-Hastings chain:

        x' = f_theta(z'),  z' ~ N(0, I)
        P_acc = min(1, [ e^{-E(x')/T} * q(x) ] / [ e^{-E(x)/T} * q(x') ] )

    This is exact w.r.t. the true LoopSage energy, but proposals are
    full-system, flow-generated jumps rather than single-anchor slides.
    """

    def __init__(self, generator: LoopSage_Generator):
        self.gen = generator
        self.energy_fn = generator.energy_fn
        self.cfg = generator.cfg
        self.device = generator.device

    def run(self, n_steps: int, x_init: Optional[torch.Tensor] = None,
            record_every: int = 1) -> dict:
        """
        Independence-sampler MH: every proposal is an i.i.d. draw from the
        flow, so it never depends on the current chain state. That means
        all n_steps proposals (plus their energies/log-probs) can be
        produced in a single batched flow pass up front, instead of one
        tiny flow call per step -- mathematically identical chain, far
        fewer tensor ops and no repeated eval()/train() toggling. The
        actual accept/reject sweep is then a cheap scalar loop over plain
        numpy arrays.
        """
        gen, energy_fn, cfg = self.gen, self.energy_fn, self.cfg

        gen.flow.eval()
        with torch.no_grad():
            if x_init is None:
                x_all, log_prob_all = gen.flow.sample(n_steps + 1, device=self.device)
                E_all = energy_fn.energy(x_all)
            else:
                x0 = x_init.to(self.device).unsqueeze(0)
                log_prob0 = gen.flow.log_prob(x0)
                E0 = energy_fn.energy(x0)
                x_prop, log_prob_prop = gen.flow.sample(n_steps, device=self.device)
                E_prop = energy_fn.energy(x_prop)
                x_all = torch.cat([x0, x_prop], dim=0)
                log_prob_all = torch.cat([log_prob0, log_prob_prop], dim=0)
                E_all = torch.cat([E0, E_prop], dim=0)
        gen.flow.train()

        # index 0 is the initial state; indices 1..n_steps are the proposals
        E_np = E_all.cpu().numpy()
        log_prob_np = log_prob_all.cpu().numpy()
        log_u = np.log(np.random.default_rng().random(n_steps))

        log.info(f"Running NeuralMCMCSampler for {n_steps} steps...")
        t0 = time.time()

        cur_idx = 0
        idx_trace = np.empty(n_steps // record_every, dtype=np.int64)
        n_accept = 0
        record_pos = 0
        log_interval = max(1, n_steps // 10)

        for step in range(1, n_steps + 1):
            prop_idx = step
            log_alpha = (-(E_np[prop_idx] - E_np[cur_idx]) / cfg.T
                         + (log_prob_np[cur_idx] - log_prob_np[prop_idx]))
            if log_u[step - 1] < min(0.0, log_alpha):
                cur_idx = prop_idx
                n_accept += 1

            if step % record_every == 0:
                idx_trace[record_pos] = cur_idx
                record_pos += 1

            if step % log_interval == 0:
                log.info(f"  step {step}/{n_steps}  acc_rate={n_accept/step:.3f}  "
                         f"E={E_np[cur_idx]:.3f}")

        elapsed = time.time() - t0
        acc_rate = n_accept / n_steps
        log.info(f"NeuralMCMC finished in {elapsed:.1f}s. Acceptance rate={acc_rate:.3f}")

        idx_t = torch.as_tensor(idx_trace, device=self.device)
        x_chain_t = x_all[idx_t]
        ms, ns = gen.cu_to_mn_discrete(x_chain_t)

        return {
            "x_chain": x_chain_t.cpu().numpy(),
            "energy": E_np[idx_trace],
            "acc_rate": acc_rate,
            "ms": ms,
            "ns": ns,
        }


# ============================================================================
# 5. Visualizations (publication-style, light background, minimal text)
# ============================================================================

def _pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
        "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": "#e5e7eb", "grid.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def plot_training_curves(history: dict, path: Optional[str] = None):
    _pub_style()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    axs[0].plot(history["kl_loss"], color="#2563eb", lw=1.3)
    axs[0].set_title("Reverse-KL loss")
    axs[0].set_xlabel("Training step")
    axs[0].set_ylabel("Loss")

    axs[1].plot(history["mean_energy"], color="#16a34a", lw=1.3)
    axs[1].set_title("Mean sampled energy ⟨E⟩")
    axs[1].set_xlabel("Training step")

    fig.suptitle("LoopSage_Generator — training diagnostics", fontsize=13, y=1.03)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_training_curves.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def plot_loop_length_comparison(ms_flow: np.ndarray, ns_flow: np.ndarray,
                                 ms_mcmc: Optional[np.ndarray] = None,
                                 ns_mcmc: Optional[np.ndarray] = None,
                                 path: Optional[str] = None):
    _pub_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    L_flow = np.abs(ns_flow - ms_flow).flatten()
    ax.hist(L_flow, bins=40, alpha=0.55, density=True, color="#2563eb",
            label="Flow samples (approx.)")

    if ms_mcmc is not None:
        L_mcmc = np.abs(ns_mcmc - ms_mcmc).flatten()
        ax.hist(L_mcmc, bins=40, alpha=0.55, density=True, color="#dc2626",
                label="Neural-MCMC (exact)")

    ax.set_xlabel("Loop length (beads)")
    ax.set_ylabel("Density")
    ax.set_title("Loop-length distribution: flow vs. MH-corrected")
    ax.legend(frameon=False)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_loop_length_comparison.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def plot_generated_heatmap(ms: np.ndarray, ns: np.ndarray, N_beads: int,
                            path: Optional[str] = None, title: str = "Generated contact map"):
    _pub_style()

    left = np.clip(np.minimum(ms, ns).astype(np.int64), 0, N_beads - 1)
    right = np.clip(np.maximum(ms, ns).astype(np.int64), 0, N_beads - 1)

    l_flat = left.ravel()
    r_flat = right.ravel()
    b = np.arange(N_beads, dtype=np.float32)

    loop_len = np.maximum(r_flat - l_flat, 1).astype(np.float32)
    sigma = loop_len / 3.0
    center = (l_flat + r_flat) / 2.0

    mat = np.zeros((N_beads, N_beads), dtype=np.float64)
    chunk = 4096
    for start in range(0, l_flat.size, chunk):
        sl = slice(start, start + chunk)
        l_c, r_c = l_flat[sl], r_flat[sl]
        c_c, s_c = center[sl], sigma[sl]
        w = np.exp(-0.5 * ((b[None, :] - c_c[:, None]) / s_c[:, None]) ** 2).astype(np.float32)
        w *= ((b[None, :] >= l_c[:, None]) & (b[None, :] <= r_c[:, None]))
        mat += w.T @ w

    mat /= max(l_flat.size, 1)

    hic_red = mcolors.LinearSegmentedColormap.from_list(
        "hic_red", ["#ffffff", "#ffcccc", "#ff4444", "#cc0000", "#6b0000"]
    )
    eps = mat[mat > 0].min() * 0.01 if mat.any() else 1e-6
    norm = mcolors.LogNorm(vmin=eps, vmax=mat.max() + eps)

    fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=150)
    im = ax.imshow(mat, cmap=hic_red, norm=norm, origin="upper", aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Contact frequency", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Bead index")
    ax.set_ylabel("Bead index")
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", f"bg_{title.lower().replace(' ', '_')}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)
    return mat


def plot_energy_trace(energies: np.ndarray, path: Optional[str] = None):
    _pub_style()
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    ax.plot(energies, color="#111111", lw=0.8)
    ax.set_xlabel("Recorded MCMC step")
    ax.set_ylabel("Energy")
    ax.set_title("Neural-MCMC energy trace")
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_mcmc_energy_trace.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def plot_energy_breakdown(history: dict, path: Optional[str] = None):
    """
    Training-time breakdown of the LoopSage Hamiltonian into its physical
    terms, plus the hard crossing count -- the same quantities shown live
    in the training progress bar, plotted over the full run.
    """
    _pub_style()
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)

    panels = [
        (axs[0, 0], "E_fold", "Folding energy ⟨E_fold⟩", "#2563eb"),
        (axs[0, 1], "E_bind", "CTCF-binding energy ⟨E_bind⟩", "#7c3aed"),
        (axs[1, 0], "E_cross", "Crossing energy ⟨E_cross⟩", "#dc2626"),
        (axs[1, 1], "n_crossings", "Crossing count ⟨#crossings⟩", "#16a34a"),
    ]
    for ax, key, title, color in panels:
        ax.plot(history[key], color=color, lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("Training step")

    fig.suptitle("LoopSage_Generator — energy-term breakdown", fontsize=13, y=1.02)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_energy_breakdown.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def plot_lef_trajectories(ms: np.ndarray, ns: np.ndarray, N_beads: int,
                           path: Optional[str] = None, max_points: int = 2000,
                           title: str = "LEF trajectories",
                           L: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
                           motif_threshold: float = 0.3):
    """
    Kymograph-style view of the sampled ensemble: each LEF's loop span
    (m_i, n_i) drawn as a translucent band across the sample/step axis, so
    overlapping loops darken and drift/extrusion patterns are visible.

    ms, ns : (n_samples, N_lef) -- typically the recorded Neural-MCMC chain,
             where the sample axis has a genuine sequential/"time" meaning.
    L, R   : optional CTCF binding tracks. When given, bead positions where
             L/R exceed `motif_threshold` * their max are drawn as thin
             horizontal reference lines, so it's visible at a glance whether
             anchors actually accumulate at CTCF motifs.
    """
    _pub_style()

    n_samples, N_lef = ms.shape
    stride = max(1, n_samples // max_points)
    ms_s, ns_s = ms[::stride], ns[::stride]
    steps = np.arange(0, n_samples, stride)

    cmap = plt.get_cmap("turbo", N_lef)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    if L is not None and L.max() > 0:
        for site in np.flatnonzero(L > motif_threshold * L.max()):
            ax.axhline(site, color="#16a34a", lw=0.6, alpha=0.35, linestyle="--", zorder=0)
    if R is not None and R.max() > 0:
        for site in np.flatnonzero(R > motif_threshold * R.max()):
            ax.axhline(site, color="#7c3aed", lw=0.6, alpha=0.35, linestyle=":", zorder=0)

    for i in range(N_lef):
        ax.fill_between(steps, ms_s[:, i], ns_s[:, i],
                         color=cmap(i), alpha=0.18, linewidth=0)
        ax.plot(steps, ms_s[:, i], color=cmap(i), lw=0.5, alpha=0.7)
        ax.plot(steps, ns_s[:, i], color=cmap(i), lw=0.5, alpha=0.7)

    if L is not None or R is not None:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color="#16a34a", lw=1.2, linestyle="--", label="Left-motif (L) site"),
                   Line2D([0], [0], color="#7c3aed", lw=1.2, linestyle=":", label="Right-motif (R) site")]
        ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)

    ax.set_xlabel("Sample / MCMC step")
    ax.set_ylabel("Bead index")
    ax.set_ylim(0, N_beads - 1)
    ax.set_title(title)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", f"bg_{title.lower().replace(' ', '_')}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def plot_average_loop_size(ms: np.ndarray, ns: np.ndarray, path: Optional[str] = None):
    """
    Mean (+/- std across LEFs) loop size over the sampled ensemble, e.g. to
    check the Neural-MCMC chain has equilibrated (no residual drift/trend).
    """
    _pub_style()

    loop_len = ns.astype(np.float64) - ms.astype(np.float64)
    mean_len = loop_len.mean(axis=1)
    std_len = loop_len.std(axis=1)
    steps = np.arange(len(mean_len))

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(steps, mean_len, color="#2563eb", lw=1.3, label="mean loop size")
    ax.fill_between(steps, mean_len - std_len, mean_len + std_len,
                     color="#2563eb", alpha=0.15, linewidth=0, label="±1 std (across LEFs)")

    ax.set_xlabel("Sample / MCMC step")
    ax.set_ylabel("Loop size (beads)")
    ax.set_title("Average loop size over time")
    ax.legend(frameon=False)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_average_loop_size.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def plot_batch_diversity(history: dict, path: Optional[str] = None):
    """
    Training-time view of the two anti-mode-collapse knobs: `batch_std`
    (how spread out sampled anchors are within a batch -- should stay
    well above ~0 if the flow keeps exploring) and `T_eff` (the annealed
    sampling temperature actually used that step). A `batch_std` that
    decays to ~0 while the loss keeps improving is the signature of mode
    collapse: the flow is minimizing the loss by memorizing one low-energy
    configuration rather than covering the Boltzmann distribution.
    """
    _pub_style()
    fig, ax1 = plt.subplots(figsize=(8, 4.2), dpi=150)

    ax1.plot(history["batch_std"], color="#2563eb", lw=1.3, label="batch_std (anchor spread)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Batch std (beads)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(history["T_eff"], color="#dc2626", lw=1.1, alpha=0.8, label="T_eff (annealed)")
    ax2.set_ylabel("T_eff", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.grid(False)

    ax1.set_title("Batch diversity vs. annealed temperature")
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_batch_diversity.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def compute_loop_hierarchy(ms: np.ndarray, ns: np.ndarray) -> np.ndarray:
    """
    Loop i contains loop j if i's span strictly encloses j's span and is
    longer. A loop's order is how deep it is nested: 0 = orphan (no
    parent), 1 = child of an orphan, k = child of an order-(k-1) loop.
    Since containment is transitive, order = total number of ancestors.

    Returns order : (n_samples, N_lef) int array.
    """
    m, n = ms.astype(np.float64), ns.astype(np.float64)
    length = n - m

    m_i, m_j = m[:, :, None], m[:, None, :]
    n_i, n_j = n[:, :, None], n[:, None, :]
    len_i, len_j = length[:, :, None], length[:, None, :]

    contains = (m_i <= m_j) & (n_j <= n_i) & (len_i > len_j)  # i contains j
    order = contains.sum(axis=1)  # ancestors of j
    return order


def plot_loop_hierarchy(ms: np.ndarray, ns: np.ndarray, path: Optional[str] = None):
    """
    Single bar plot of loop order pooled across the ensemble: "Orphan" =
    no parent, "Order k" = nested k levels deep (k=1 has a parent, k=2 a
    grandparent, ...). A real extrusion hierarchy has a mix of orders, not
    everything piled at Orphan (which would mean loops are only ever
    side-by-side, never nested).
    """
    _pub_style()
    order = compute_loop_hierarchy(ms, ns).ravel()
    max_order = int(order.max())

    counts = [int((order == k).sum()) for k in range(max_order + 1)]
    labels = ["Orphan"] + [f"Order {k}" for k in range(1, max_order + 1)]
    frac = np.array(counts) / max(order.size, 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    colors = plt.get_cmap("Blues")(np.linspace(0.4, 0.9, len(counts)))
    bars = ax.bar(labels, frac, color=colors, edgecolor="white")
    for bar, f in zip(bars, frac):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{f:.0%}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Fraction of loops")
    ax.set_title("Loop nesting hierarchy")
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_loop_hierarchy.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def run_robustness_checks(generator: "LoopSage_Generator", result: dict,
                           min_loop_length: float, print_report: bool = True) -> dict:
    """
    Short pass/warn sanity report, run once after training + sampling:
    (1) loops actually extrude past minimal length, (2) the reverse-KL
    loss went down and stayed numerically stable, (3) the batch didn't
    mode-collapse, (4) the sampled ensemble contains nested loop families
    (not just flat orphans), (5) the Neural-MCMC acceptance rate is high
    enough to trust the flow. Returns the same numbers as a dict.
    """
    hist = generator.history
    ms, ns = result["ms"], result["ns"]
    lengths = ns - ms

    checks = {}

    frac_stuck = float((lengths <= min_loop_length + 1).mean())
    checks["extrusion"] = dict(ok=frac_stuck < 0.5, frac_stuck=frac_stuck,
                                mean_length=float(lengths.mean()))

    loss = np.array(hist["kl_loss"])
    n_skip = len(hist["skipped_steps"])
    if len(loss) >= 20:
        k = max(1, len(loss) // 10)
        early, late = float(loss[:k].mean()), float(loss[-k:].mean())
    else:
        early, late = float("nan"), float("nan")
    checks["training"] = dict(ok=(late < early) and n_skip < 0.05 * max(len(loss), 1),
                               loss_early=early, loss_late=late, n_skipped=n_skip)

    batch_std = np.array(hist["batch_std"])
    late_std = float(batch_std[-max(1, len(batch_std) // 10):].mean()) if len(batch_std) else 0.0
    checks["diversity"] = dict(ok=late_std > 1.0, batch_std=late_std)

    order = compute_loop_hierarchy(ms, ns)
    frac_nested = float((order > 0).mean())
    checks["families"] = dict(ok=frac_nested > 0.05, frac_nested=frac_nested)

    acc = result.get("acc_rate")
    if acc is not None:
        checks["mcmc"] = dict(ok=acc > 0.05, acc_rate=float(acc))

    if print_report:
        def tag(ok):
            return "OK  " if ok else "WARN"
        print("\n=== Robustness checks ===")
        e = checks["extrusion"]
        print(f"[{tag(e['ok'])}] extrusion: {e['frac_stuck']:.0%} of loops near minimal length "
              f"(mean length={e['mean_length']:.1f}, min={min_loop_length:.0f})")
        t = checks["training"]
        print(f"[{tag(t['ok'])}] training: loss {t['loss_early']:.3f} -> {t['loss_late']:.3f}, "
              f"{t['n_skipped']} skipped non-finite steps")
        d = checks["diversity"]
        print(f"[{tag(d['ok'])}] diversity: batch_std={d['batch_std']:.2f} late in training "
              f"(near 0 = collapsed onto one arrangement)")
        f = checks["families"]
        print(f"[{tag(f['ok'])}] families: {f['frac_nested']:.0%} of loops are nested inside "
              f"another loop (0% = only flat, side-by-side loops)")
        if acc is not None:
            m = checks["mcmc"]
            print(f"[{tag(m['ok'])}] flow quality: Neural-MCMC acceptance rate={m['acc_rate']:.1%}")
        print("==========================\n")

    return checks


def autocorrelation(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Normalized autocorrelation of a 1D series, lags 0..max_lag (acf[0]=1)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = (x ** 2).sum()
    if var <= 0:
        return np.zeros(max_lag + 1)
    max_lag = min(max_lag, len(x) - 1)
    acf = np.array([(x[:len(x) - k] * x[k:]).sum() / var for k in range(max_lag + 1)])
    return acf


def plot_autocorrelation(result: dict, path: Optional[str] = None, max_lag: int = 200):
    """
    Autocorrelation of the Neural-MCMC chain (energy, mean loop size).
    Fast decay to ~0 = consecutive samples are effectively independent
    (expected for a good independence sampler); slow decay = poor mixing.
    """
    _pub_style()
    energy_acf = autocorrelation(result["energy"], max_lag)
    mean_len = (result["ns"].astype(np.float64) - result["ms"].astype(np.float64)).mean(axis=1)
    len_acf = autocorrelation(mean_len, max_lag)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    lags = np.arange(len(energy_acf))
    ax.plot(lags, energy_acf, color="#dc2626", lw=1.4, label="Total energy")
    ax.plot(lags, len_acf, color="#2563eb", lw=1.4, label="Mean loop size")
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Neural-MCMC chain autocorrelation")
    ax.legend(frameon=False)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_autocorrelation.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def plot_anchor_ctcf_alignment(ms: np.ndarray, ns: np.ndarray, L: np.ndarray, R: np.ndarray,
                                N_beads: int, path: Optional[str] = None):
    """
    Where do anchors actually land? Histograms of left/right anchor
    positions overlaid on the L/R motif tracks -- peaks lining up means
    LEFs are capturing CTCF sites in the right orientation.
    """
    _pub_style()
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), dpi=150, sharex=True)

    bins = np.linspace(0, N_beads - 1, min(N_beads, 150))

    axs[0].hist(ms.ravel(), bins=bins, color="#2563eb", alpha=0.6, density=True,
                label="Left-anchor density (m)")
    ax0b = axs[0].twinx()
    ax0b.plot(np.arange(N_beads), L, color="#16a34a", lw=1.2, label="L (left-motif strength)")
    ax0b.set_ylabel("L", color="#16a34a")
    ax0b.tick_params(axis="y", labelcolor="#16a34a")
    ax0b.grid(False)
    axs[0].set_ylabel("Density")
    axs[0].set_title("Left anchors vs. left-CTCF-motif track")

    axs[1].hist(ns.ravel(), bins=bins, color="#dc2626", alpha=0.6, density=True,
                label="Right-anchor density (n)")
    ax1b = axs[1].twinx()
    ax1b.plot(np.arange(N_beads), R, color="#16a34a", lw=1.2, label="R (right-motif strength)")
    ax1b.set_ylabel("R", color="#16a34a")
    ax1b.tick_params(axis="y", labelcolor="#16a34a")
    ax1b.grid(False)
    axs[1].set_ylabel("Density")
    axs[1].set_xlabel("Bead index")
    axs[1].set_title("Right anchors vs. right-CTCF-motif track")

    fig.suptitle("Anchor-CTCF alignment (convergent-motif capture check)", fontsize=13, y=1.02)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_anchor_ctcf_alignment.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def plot_loop_arcs(ms: np.ndarray, ns: np.ndarray, N_beads: int,
                    path: Optional[str] = None, n_samples: int = 4, seed: int = 0):
    """
    Classic arc diagram: each loop drawn as a semicircle from m to n,
    filled light green, beads on the x-axis. One row per sampled snapshot,
    so you can see the loop pattern (and whether it changes) sample to
    sample -- the direct, per-configuration picture that a kymograph
    averages away.
    """
    _pub_style()
    rng = np.random.default_rng(seed)
    n_total = ms.shape[0]
    idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    idx.sort()

    fig, axs = plt.subplots(len(idx), 1, figsize=(9, 2.0 * len(idx)), dpi=150, sharex=True)
    if len(idx) == 1:
        axs = [axs]

    theta = np.linspace(0, np.pi, 60)
    for ax, s in zip(axs, idx):
        for m, n in zip(ms[s], ns[s]):
            if n <= m:
                continue
            r = (n - m) / 2.0
            xs = m + r * (1 - np.cos(theta))
            ys = r * np.sin(theta)
            ax.fill(xs, ys, color="#86efac", alpha=0.5, edgecolor="#16a34a", linewidth=0.6)
        ax.set_xlim(0, N_beads - 1)
        ax.set_yticks([])
        ax.set_ylabel(f"sample {s}", fontsize=9, rotation=0, ha="right", va="center")
        ax.grid(False)

    axs[-1].set_xlabel("Bead index")
    fig.suptitle("Loop anchors as arcs", fontsize=13, y=1.0)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        out = os.path.join(path, "plots", "bg_loop_arcs.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


# ============================================================================
# 6. Runnable example — every hyperparameter defined and explained right here
# ============================================================================

def main():
    """
    End-to-end example:
      1. Build L/R binding potentials from a .bedpe file (reusing preproc.py)
      2. Train a LoopSage_Generator (Boltzmann Generator) on the pure
         folding+crossing+binding energy
      3. Save the trained model
      4. Use NeuralMCMCSampler to generate an exact ensemble
      5. Produce diagnostic + heatmap visualizations

    No CLI/argparse: every hyperparameter is a plain variable below, grouped
    and commented so it's all visible in one place. Edit the values directly
    to change a run.
    """

    # ---------------------------------------------------------------------
    # Runtime
    # ---------------------------------------------------------------------
    device = "cpu"          # "cpu" (default), or "cuda" / "cuda:0" etc. if you have a GPU.
                             # Falls back to CPU automatically (with a warning) if CUDA
                             # is requested but not actually available.
    seed = 0                 # Random seed for reproducibility (None to disable).

    # ---------------------------------------------------------------------
    # Input data / genomic region
    # ---------------------------------------------------------------------
    interaction_file = "/home/blackpianocat/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10.bedpe"
    chrom = "chr6"
    region = [15_550_000, 16_050_000]   # [start, end] in bp
    out_dir = "bg_results"              # trained model, plots and logs are written here
    smooth_sigma = 8.0                  # Gaussian smoothing (beads) applied to the L/R CTCF tracks.
                                          # Too narrow (e.g. 2) leaves most of the polymer with exactly
                                          # zero binding gradient between peaks, so only LEFs that land
                                          # right on a motif ever feel a pull -- the rest have nothing
                                          # but the (identical, symmetric) folding term to act on, which
                                          # can produce a "few big winners, rest stuck small" ensemble.
                                          # Wider smoothing gives more LEFs a felt gradient toward the
                                          # nearest motif. Raise further if that pattern persists.

    # ---------------------------------------------------------------------
    # LoopSage Hamiltonian (physical) hyperparameters
    #   E = f * E_fold + b * E_bind + kappa * E_cross   (see module docstring)
    # ---------------------------------------------------------------------
    N_beads = 500            # polymer resolution: number of beads spanning `region`
    N_lef = 40                # number of loop-extruding factors (e.g. cohesin) to model
    f = 2.0                    # folding coefficient: strength of the entropic loop-length term
    b = 1.0                    # CTCF-binding coefficient: strength of anchor attraction to L/R sites
    kappa = 1.0                # crossing coefficient: penalty strength against LEF crossing/overlap
    T = 1.0                    # Boltzmann temperature the generator is trained to sample at
    crossing_softness = 1.0    # softness (beads) of the differentiable surrogate for the discrete
                                # crossing indicator Kappa(); smaller = sharper/more exact
    crossing_safety = 1.0     # how many multiples of the whole system's fold/bind energy scale
                                # a single fully-crossing pair costs (at kappa=1); keeps the
                                # crossing penalty a strong, near-hard constraint at any N_beads
                                # instead of a size-independent fixed constant
    min_loop_length = 4.0      # hard lower bound (beads) on loop length n-m
    max_loop_length = None     # hard upper bound (beads) on loop length n-m; None -> N_beads - 1
                                # (these two bounds are what keep the generator numerically
                                # stable by construction -- see the module docstring)
    rectify_binding = True     # clip preproc's contrastive-normalized L/R at 0 so "no CTCF here"
                                # is exactly 0 (not a stray negative that flips positive through
                                # bind_norm<0) -- this is what keeps E_bind <= 0 everywhere; see
                                # LoopSageEnergyConfig.rectify_binding for the full explanation
    fold_mode = "log"          # how loop length enters the folding term: "log" (default, matches
                                # stochastic_simulation.py, diminishing reward for growth), "linear"
                                # (constant reward per bead), or "quadratic" (reward grows with
                                # length -- pushes hardest once a loop is already extruding).

    # ---------------------------------------------------------------------
    # Normalizing-flow architecture
    # ---------------------------------------------------------------------
    n_layers = 8                # number of RealNVP coupling layers
    hidden = 128                 # hidden width of each coupling-layer MLP
    max_log_scale = 2.0          # per-layer bound on |log_s|; larger = more expressive flow
                                  # but higher risk of unstable scale factors
    equivariant_flow = True      # use a permutation-equivariant conditioner (Deep-Sets style) so the
                                  # flow treats the N_lef LEFs as an interchangeable set, matching the
                                  # energy's actual symmetry -- a plain per-index MLP conditioner
                                  # privileges specific LEF slots, which biases which ones the flow
                                  # finds it easy to grow vs. leave stuck. False uses the old MLP.

    # ---------------------------------------------------------------------
    # Training (pure reverse-KL against the LoopSage energy)
    # ---------------------------------------------------------------------
    n_steps = 4000               # number of reverse-KL training steps
    batch_size = 256              # samples per training step
    lr = 1e-3                     # Adam learning rate
    grad_clip = 5.0                # gradient-norm clipping value
    log_every = 200                 # detailed log-line frequency (steps); the progress bar itself
                                     # updates every step with the current loss / mean energy
    kappa_warmup_steps = 500         # linearly ramp the crossing-penalty weight from 0 to full
                                      # strength over this many steps, so folding/binding can shape
                                      # the distribution before the (much larger) crossing term
                                      # dominates -- this is what lets E_bind settle negative
                                      # (anchors actually finding CTCF sites) instead of getting
                                      # stuck positive; 0 disables the warm-up
    temp_anneal_factor = 5.0         # train at an effective temperature up to this many times T,
                                      # annealed down to T -- reverse-KL is "mode-seeking" and will
                                      # otherwise collapse onto a single low-energy configuration
                                      # (same LEF arrangement every sample, independent of T at
                                      # sampling time); a hot start forces the flow to cover more of
                                      # the landscape before it commits. 1.0 disables annealing.
                                      # Raised from 3.0 -- mode collapse was still observed at that
                                      # setting, watch plot_batch_diversity to see if this is enough.
    temp_anneal_steps = n_steps       # anneal T_eff -> T linearly over the whole run (was n_steps//2)
                                      # -- gives the flow more time to explore before committing.
    lambda_diversity = 0.05          # weight of an explicit batch-diversity regularizer that
                                      # penalizes low anchor variance within a training batch --
                                      # a second, direct defense against mode collapse alongside
                                      # temperature annealing; 0 disables it

    # ---------------------------------------------------------------------
    # Neural-MCMC sampling (exact Boltzmann ensemble)
    # ---------------------------------------------------------------------
    n_mcmc_steps = 3000            # number of Metropolis-Hastings proposals
    n_flow_samples = 2000           # number of pure (approximate) flow samples, for diagnostics

    # ---------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------
    resolved_device = resolve_device(device)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    log.info("=" * 60)
    log.info("LoopSage Boltzmann Generator — run configuration")
    log.info("=" * 60)
    for name, val in [
        ("device", device), ("resolved device", resolved_device), ("seed", seed),
        ("interaction_file", interaction_file), ("chrom", chrom), ("region", region),
        ("out_dir", out_dir), ("smooth_sigma", smooth_sigma),
        ("N_beads", N_beads), ("N_lef", N_lef), ("f", f), ("b", b), ("kappa", kappa), ("T", T),
        ("crossing_softness", crossing_softness), ("crossing_safety", crossing_safety),
        ("min_loop_length", min_loop_length), ("max_loop_length", max_loop_length),
        ("rectify_binding", rectify_binding),
        ("fold_mode", fold_mode),
        ("n_layers", n_layers), ("hidden", hidden), ("max_log_scale", max_log_scale),
        ("equivariant_flow", equivariant_flow),
        ("n_steps", n_steps), ("batch_size", batch_size), ("lr", lr), ("grad_clip", grad_clip),
        ("log_every", log_every), ("kappa_warmup_steps", kappa_warmup_steps),
        ("temp_anneal_factor", temp_anneal_factor), ("temp_anneal_steps", temp_anneal_steps),
        ("lambda_diversity", lambda_diversity),
        ("n_mcmc_steps", n_mcmc_steps), ("n_flow_samples", n_flow_samples),
    ]:
        log.info(f"  {name:20s} = {val}")
    log.info("=" * 60)

    log.info("Building L/R binding potentials from CTCF interaction file...")
    L, R, J, J_loss, stats = binding_vectors_from_bedpe(
        bedpe_file=interaction_file,
        N_beads=N_beads,
        chrom=chrom,
        region=region,
        out_path=out_dir,
        viz=False,
        smooth=True,
        smooth_sigma=smooth_sigma,
        contrastive=True,
    )

    cfg = LoopSageEnergyConfig(
        N_beads=N_beads, N_lef=N_lef,
        f=f, b=b, kappa=kappa,
        crossing_softness=crossing_softness, crossing_safety=crossing_safety, T=T,
        min_loop_length=min_loop_length, max_loop_length=max_loop_length,
        rectify_binding=rectify_binding, fold_mode=fold_mode,
    )

    log.info("Constructing LoopSage_Generator (RealNVP flow)...")
    generator = LoopSage_Generator(
        L, R, cfg,
        n_layers=n_layers, hidden=hidden, max_log_scale=max_log_scale,
        device=device, seed=seed, equivariant_flow=equivariant_flow,
    )

    # Sanity check: energy of three baseline (non-learned) configurations,
    # to confirm the Hamiltonian behaves as expected before training --
    # "parallel" and "minimal" should have E_cross~0 (no crossings by
    # construction); "random" is the un-shaped starting point training
    # should improve on.
    log.info("Baseline energies (untrained reference configurations):")
    for mode in ("parallel", "random", "minimal"):
        ms0, ns0 = generate_initial_configuration(N_beads, N_lef, mode=mode,
                                                    min_loop_length=min_loop_length, seed=seed)
        m0 = torch.as_tensor(ms0, dtype=torch.float32, device=resolved_device)
        n0 = torch.as_tensor(ns0, dtype=torch.float32, device=resolved_device)
        comp0 = generator.energy_fn.energy_from_mn(m0, n0)
        log.info(f"  {mode:10s}: E_total={comp0['E_total'].item():10.2f}  "
                 f"E_fold={comp0['E_fold'].item():8.2f}  E_bind={comp0['E_bind'].item():8.2f}  "
                 f"E_cross={comp0['E_cross'].item():8.2f}  #crossings={comp0['n_crossings'].item():.1f}")

    generator.train_flow(
        n_steps=n_steps, batch_size=batch_size, lr=lr,
        log_every=log_every, grad_clip=grad_clip,
        kappa_warmup_steps=kappa_warmup_steps,
        temp_anneal_factor=temp_anneal_factor, temp_anneal_steps=temp_anneal_steps,
        lambda_diversity=lambda_diversity,
    )

    model_path = os.path.join(out_dir, "loopsage_generator.pt")
    generator.save(model_path)

    plot_training_curves(generator.history, path=out_dir)
    plot_energy_breakdown(generator.history, path=out_dir)
    plot_batch_diversity(generator.history, path=out_dir)

    # ---- pure flow samples (approximate) ----
    log.info("Sampling directly from the trained flow (approximate)...")
    x_flow, _ = generator.sample_continuous(n_flow_samples)
    ms_flow, ns_flow = generator.cu_to_mn_discrete(x_flow)
    plot_generated_heatmap(ms_flow, ns_flow, N_beads, path=out_dir,
                            title="Flow-only heatmap")

    # ---- neural-MCMC (exact w.r.t. true energy) ----
    log.info("Running NeuralMCMCSampler for exact Boltzmann samples...")
    sampler = NeuralMCMCSampler(generator)
    result = sampler.run(n_steps=n_mcmc_steps, record_every=1)

    plot_energy_trace(result["energy"], path=out_dir)
    plot_loop_length_comparison(ms_flow, ns_flow, result["ms"], result["ns"], path=out_dir)
    plot_generated_heatmap(result["ms"], result["ns"], N_beads, path=out_dir,
                            title="Neural-MCMC heatmap")
    plot_lef_trajectories(result["ms"], result["ns"], N_beads, path=out_dir, L=L, R=R)
    plot_average_loop_size(result["ms"], result["ns"], path=out_dir)
    plot_loop_hierarchy(result["ms"], result["ns"], path=out_dir)
    plot_autocorrelation(result, path=out_dir)
    plot_anchor_ctcf_alignment(result["ms"], result["ns"], L, R, N_beads, path=out_dir)
    plot_loop_arcs(result["ms"], result["ns"], N_beads, path=out_dir)

    run_robustness_checks(generator, result, min_loop_length=min_loop_length)

    log.info(f"Acceptance rate of neural MCMC: {result['acc_rate']:.3f}")
    log.info(f"Done. All outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()