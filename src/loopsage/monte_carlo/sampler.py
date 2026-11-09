"""Bundle of everything the (numba) sampler needs, plus normalisation constants.

- Monte Carlo proposals for one LEF: rebinding and sliding (optionally at LEF-specific speeds).
- Metropolis / simulated-annealing sampler (compiled with numba).
"""
from collections import namedtuple

import numpy as np
from numba import njit

from loopsage.monte_carlo.energy import dE_edge, dE_epi, dE_epi_dyn, E_bind_inner, E_cross, E_data, fold_energy


# ---- Bundle of everything the (numba) sampler needs, plus normalisation constants ----

# Energy terms are scaled so coefficients (f, b, kappa) are comparable across systems.
MCParams = namedtuple("MCParams", [
    "N_beads", "N_lef", "N_lef2",
    "fold_norm", "fold_norm2", "bind_norm", "k_norm", "epi_norm",
    "cross_loop", "between", "loss_mode", "epi_dynamic", "p_epigenetic", "n_epi", "step_length", "rw",
    "annealing", "T", "T_min",
    "L", "R", "J", "J_loss", "h", "BWs", "r", "use_track", "cdf", "use_speed", "speed_cdf", "P", "cov_norm",
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
    if len(r) != n_bw:
        raise ValueError(f"bw_strengths has {len(r)} values but there are {n_bw} tracks")
    track = inputs.lef_track
    step = int(lo.loop_lengths.min()) if cfg.variable_sliding_step and len(lo.loop_lengths) else 1
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    h = inputs.comp if inputs.comp is not None else np.zeros(n_beads)

    bind_norm = -2 * n_beads * cfg.bind_coeff / lo.n_sites
    cover = lo.cover is not None and cfg.binding_mode == "contrastive"
    P = np.r_[0.0, np.cumsum(lo.cover)] if cover else np.zeros(0)           # prefix sums: the coverage of a loop's beads in O(1)
    cov_norm = bind_norm / max(float(np.mean(lo.loop_lengths)), 1.0) if cover else 0.0     # per bead: a mean-length loop in a dense region ~ one anchor

    return MCParams(
        n_beads, n_lef, cfg.n_lef2,
        fold * cfg.folding_coeff, fold * cfg.folding_coeff2, bind_norm,
        cfg.cross_coeff * 1e4, float(cfg.epi_coeff),
        bool(cfg.cross_loop), bool(cfg.between_families_penalty), int(cfg.data_loss_mode), cfg.epigenetic_mode == "dynamic",
        float(cfg.p_epigenetic), int(cfg.n_epi_states), max(step, 1), bool(cfg.lef_rw),
        cfg.method == "Annealing", float(cfg.t_init), float(cfg.t_final),
        f64(lo.L), f64(lo.R), f64(lo.J), f64(lo.J_loss), f64(h), f64(inputs.BWs).reshape(n_bw, n_beads), f64(r),
        track is not None, f64(np.cumsum(track / track.sum())) if track is not None else np.zeros(0),
        cfg.lef_speed_dist != "constant", np.zeros(0),          # speed_cdf is set per chain (see lef_speeds)
        P, cov_norm,
    )


def lef_speeds(cfg, n: int, rng) -> np.ndarray:
    """Relative speed (sliding rate) of each LEF, drawn from the chosen distribution; mean is exactly 1."""
    kind = cfg.lef_speed_dist
    if cfg.lef_speed_spread < 0 or not 0 <= cfg.lef_fast_fraction <= 1 or cfg.lef_fast_ratio <= 0:
        raise ValueError("lef_speed_spread must be >= 0, lef_fast_fraction in [0, 1] and lef_fast_ratio > 0")
    if kind == "maxwell":                       # |v| of a 3D Gaussian velocity, scale a: mean speed = 2a*sqrt(2/pi)
        v = np.linalg.norm(rng.normal(0, np.sqrt(np.pi / 8), (n, 3)), axis=1)
    elif kind == "lognormal":
        v = np.exp(cfg.lef_speed_spread * rng.normal(size=n))
    elif kind == "two_state":
        v = np.where(rng.random(n) < cfg.lef_fast_fraction, cfg.lef_fast_ratio, 1.0)
    else:
        return np.ones(n)
    return v / v.mean()


# ---- Monte Carlo proposals for one LEF: rebinding and sliding --------------

@njit(cache=True)
def rebind(p):
    """Unbind and load a new minimal loop (m, m+2), optionally at preferred loading sites."""
    if p.use_track:
        m = min(np.searchsorted(p.cdf, np.random.rand()), p.N_beads - 3)
    else:
        m = np.random.randint(0, p.N_beads - 3)
    return m, m + 2


@njit(cache=True)
def slide(p, m_old, n_old, ms, ns):
    """Sliding step of the original LoopSage: each leg moves +-1..step_length beads (random direction).

    Same order of checks as the original: bounds, minimal loop size, then a leg that would land on
    another loop's opposite leg stays where it was. With rw off, legs only move outward (extrusion).
    """
    dm, dn = 1, 1
    if p.step_length > 1:
        dm, dn = 1 + np.random.randint(0, p.step_length), 1 + np.random.randint(0, p.step_length)
    s1, s2 = -1, 1
    if p.rw:
        s1 = -1 if np.random.rand() < 0.5 else 1
        s2 = -1 if np.random.rand() < 0.5 else 1
    m_new = max(0, m_old + s1 * dm)
    n_new = min(p.N_beads - 1, n_old + s2 * dn)
    if n_new <= m_new + 1:
        return m_old, n_old
    for k in range(ms.shape[0]):
        if ns[k] == m_new:
            m_new = m_old
    for k in range(ms.shape[0]):
        if ms[k] == n_new:
            n_new = n_old
    return m_new, n_new


# ---- Metropolis / simulated-annealing sampler (compiled with numba) --------

@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def init_state(p):
    """Random minimal loops + random epigenetic states (only if the epigenetic model is on)."""
    n_tot = p.N_lef + p.N_lef2
    ms, ns = np.zeros(n_tot, np.int64), np.zeros(n_tot, np.int64)
    for i in range(n_tot):
        ms[i], ns[i] = rebind(p)
    S = np.zeros(p.N_beads, np.int64)
    if p.epi_norm != 0.0:
        for b in range(p.N_beads):
            S[b] = np.random.randint(0, p.n_epi)
    return ms, ns, S


@njit(cache=True)
def _accept(dE, T):
    return dE <= 0 or np.exp(-dE / T) > np.random.rand()


@njit(cache=True, nogil=True)      # nogil: replicas run in parallel Python threads
def mc_block(p, i0, i1, n_steps, mc_step, ms, ns, S, E, rec, acc):
    """Run steps [i0, i1); record a snapshot whenever step % mc_step == 0.

    rec = (Ms, Ns, Es, Ks, Fs, Bs, ufs, Ts, epi) output arrays; acc[move] = (proposed, accepted)
    with move 0 = rebind, 1 = slide, 2 = epigenetic-state change. Returns the updated energy.
    """
    Ms, Ns, Es, Ks, Fs, Bs, ufs, Ts, epi = rec
    n_tot = p.N_lef + p.N_lef2
    for i in range(i0, i1):
        T = p.T - (p.T - p.T_min) * (i + 1) / n_steps if p.annealing else p.T
        for _ in range(2 * n_tot):
            if p.epi_norm != 0.0 and np.random.rand() < p.p_epigenetic:
                k, s_new = np.random.randint(0, p.N_beads), np.random.randint(0, p.n_epi)
                if s_new == S[k]:
                    s_new = (S[k] + 1) % p.n_epi
                if p.epi_dynamic:
                    dE = dE_epi_dyn(S, ms, ns, p.h, p.epi_norm, k, s_new)
                else:
                    dE = dE_epi(S, p.J, p.h, p.epi_norm, k, s_new)
                acc[2, 0] += 1
                if _accept(dE, T):
                    S[k] = s_new; E += dE; acc[2, 1] += 1
            else:
                if p.use_speed:                      # slides pick LEFs in proportion to their speed
                    kind = np.random.randint(0, 2)
                    j = min(np.searchsorted(p.speed_cdf, np.random.rand()), n_tot - 1) if kind == 1 else np.random.randint(0, n_tot)
                else:
                    j, kind = np.random.randint(0, n_tot), np.random.randint(0, 2)
                if kind == 0:
                    m_new, n_new = rebind(p)
                else:
                    m_new, n_new = slide(p, ms[j], ns[j], ms, ns)
                dE = dE_edge(p, ms, ns, S, m_new, n_new, j)
                acc[kind, 0] += 1
                if _accept(dE, T):
                    ms[j], ns[j] = m_new, n_new; E += dE; acc[kind, 1] += 1
        if i % mc_step == 0 and i // mc_step < Es.shape[0]:
            k = i // mc_step
            Es[k], Ts[k] = E, T
            Ks[k] = E_cross(ms, ns, p.k_norm, p.N_lef, p.cross_loop, p.between)
            Fs[k] = fold_energy(p, ms, ns)
            Bs[k] = E_data(p.loss_mode, p.L, p.R, p.J_loss, ms, ns, p.bind_norm)
            if p.P.shape[0] > 0:
                Bs[k] += E_bind_inner(p.P, p.cov_norm, ms, ns)
            covered = np.zeros(p.N_beads)
            for j in range(n_tot):
                covered[ms[j]:ns[j]] = 1.0
            ufs[k] = 2.0 * (p.N_beads - np.count_nonzero(covered)) / p.N_beads
            Ms[:, k], Ns[:, k], epi[:, k] = ms, ns, S
    return E
