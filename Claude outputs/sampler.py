"""Metropolis / simulated-annealing sampler (compiled with numba).

One MC step = 2 x N_lef proposals. Each proposal is a LEF move (rebind or slide)
or, if the epigenetic coupling is on, a Potts-state flip.
"""
import numpy as np
from numba import njit

from monte_carlo.energy.crossing import E_cross
from monte_carlo.energy.data_terms import E_data
from monte_carlo.energy.epigenetic import dE_epi
from monte_carlo.moves import rebind, slide
from monte_carlo.energy.total import dE_edge
from monte_carlo.energy.folding import fold_energy


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def init_state(p):
    """Random minimal loops + random epigenetic states (only if Potts is on)."""
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


@njit(cache=True)
def mc_block(p, i0, i1, n_steps, mc_step, ms, ns, S, E, rec, acc):
    """Run steps [i0, i1); record a snapshot whenever step % mc_step == 0.

    rec = (Ms, Ns, Es, Ks, Fs, Bs, ufs, Ts, epi) output arrays; acc[move] = (proposed, accepted)
    with move 0 = rebind, 1 = slide, 2 = state flip. Returns the updated energy.
    """
    Ms, Ns, Es, Ks, Fs, Bs, ufs, Ts, epi = rec
    n_tot = p.N_lef + p.N_lef2
    for i in range(i0, i1):
        T = p.T - (p.T - p.T_min) * (i + 1) / n_steps if p.annealing else p.T
        for _ in range(2 * n_tot):
            if p.epi_norm != 0.0 and np.random.rand() < p.p_spin:
                k, s_new = np.random.randint(0, p.N_beads), np.random.randint(0, p.n_epi)
                if s_new == S[k]:
                    s_new = (S[k] + 1) % p.n_epi
                dE = dE_epi(S, p.J, p.h, p.epi_norm, k, s_new)
                acc[2, 0] += 1
                if _accept(dE, T):
                    S[k] = s_new; E += dE; acc[2, 1] += 1
            else:
                j, kind = np.random.randint(0, n_tot), np.random.randint(0, 2)
                if kind == 0:
                    m_new, n_new = rebind(p)
                else:
                    m_new, n_new = slide(p, ms[j], ns[j], ms, ns)
                dE = dE_edge(p, ms, ns, m_new, n_new, j)
                acc[kind, 0] += 1
                if _accept(dE, T):
                    ms[j], ns[j] = m_new, n_new; E += dE; acc[kind, 1] += 1
        if i % mc_step == 0 and i // mc_step < Es.shape[0]:
            k = i // mc_step
            Es[k], Ts[k] = E, T
            Ks[k] = E_cross(ms, ns, p.k_norm, p.N_lef, p.cross_loop, p.between)
            Fs[k] = fold_energy(p, ms, ns)
            Bs[k] = E_data(p.loss_mode, p.L, p.R, p.J_loss, ms, ns, p.bind_norm)
            covered = np.zeros(p.N_beads)
            for j in range(n_tot):
                covered[ms[j]:ns[j]] = 1.0
            ufs[k] = 2.0 * (p.N_beads - np.count_nonzero(covered)) / p.N_beads
            Ms[:, k], Ns[:, k], epi[:, k] = ms, ns, S
    return E
