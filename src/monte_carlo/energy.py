"""Crossing energy: penalty for loops that cross or share anchors.

- Data (binding) energy: how well loop anchors match the experimental signal.
- Folding energy: rewards long loops (norm < 0), logarithmically.
- Extra binding energy from BigWig tracks (e.g. RNAPII, other ChIP-seq).
- Epigenetic-model energy for states S_i on beads (static or dynamic coupling J).
- Total energy and single-LEF energy changes, from the parameter bundle.
"""
import numpy as np
from numba import njit


# ---- Crossing energy: penalty for loops that cross or share anchors --------

@njit(cache=True)
def Kappa(mi, ni, mj, nj, cross_loop=True):
    """Crossing count of loops (mi, ni) and (mj, nj): interleaved or sharing an anchor."""
    k = 0.0
    if cross_loop:
        if mi < mj < ni < nj: k += 1
        if mj < mi < nj < ni: k += 1
    if mj == ni or mi == nj or ni == nj or mi == mj: k += 1
    return k


@njit(cache=True)
def _counts(i, j, N_lef, between):
    """Is the pair (i, j) penalised? Same family always; across families if `between`."""
    return between or (i < N_lef and j < N_lef) or (i >= N_lef and j >= N_lef)


@njit(cache=True)
def E_cross(ms, ns, k_norm, N_lef, cross_loop=True, between=True):
    """Total crossing energy over all LEF pairs."""
    total = 0.0
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            if _counts(i, j, N_lef, between):
                total += Kappa(ms[i], ns[i], ms[j], ns[j], cross_loop)
    return k_norm * total


@njit(cache=True)
def dE_cross(ms, ns, m_new, n_new, idx, k_norm, N_lef, cross_loop=True, between=True):
    """Energy change when LEF `idx` moves to (m_new, n_new)."""
    k_old, k_new = 0.0, 0.0
    for i in range(len(ms)):
        if i != idx and _counts(idx, i, N_lef, between):
            k_old += Kappa(ms[idx], ns[idx], ms[i], ns[i], cross_loop)
            k_new += Kappa(m_new, n_new, ms[i], ns[i], cross_loop)
    return k_norm * (k_new - k_old)


# ---- Data (binding) energy: how well loop anchors match the experimental signal ----

@njit(cache=True)
def E_bind(L, R, ms, ns):
    """1D: sum of L at left legs and R at right legs."""
    return np.sum(L[ms]) + np.sum(R[ns])


@njit(cache=True)
def E_2d(J_loss, ms, ns):
    """2D: symmetrised loop-map value at every (m, n)."""
    E = 0.0
    for k in range(ms.shape[0]):
        E += 0.5 * (J_loss[ms[k], ns[k]] + J_loss[ns[k], ms[k]])
    return E


@njit(cache=True)
def E_data(mode, L, R, J_loss, ms, ns, bind_norm):
    """mode 0 = 1D anchors, 1 = 2D loop map."""
    return bind_norm * (E_bind(L, R, ms, ns) if mode == 0 else E_2d(J_loss, ms, ns))


@njit(cache=True)
def dE_data(mode, L, R, J_loss, bind_norm, m_old, n_old, m_new, n_new):
    """Change of E_data when one LEF moves (m_old, n_old) -> (m_new, n_new)."""
    if mode == 0:
        return bind_norm * (L[m_new] + R[n_new] - L[m_old] - R[n_old])
    new = 0.5 * (J_loss[m_new, n_new] + J_loss[n_new, m_new])
    old = 0.5 * (J_loss[m_old, n_old] + J_loss[n_old, m_old])
    return bind_norm * (new - old)


# ---- Interior part of the binding energy (contrastive mode): loops are rewarded for spanning beads that the input loops cover ----

@njit(cache=True)
def E_bind_inner(P, cov_norm, ms, ns):
    """cov_norm * sum over loops of the coverage summed over the loop's beads (P = prefix sums of the coverage profile)."""
    return cov_norm * np.sum(P[ns] - P[ms])


@njit(cache=True)
def dE_bind_inner(P, cov_norm, m_old, n_old, m_new, n_new):
    return cov_norm * ((P[n_new] - P[m_new]) - (P[n_old] - P[m_old]))


# ---- Folding energy: rewards long loops (norm < 0), logarithmically --------

@njit(cache=True)
def E_fold(ms, ns, fold_norm):
    return fold_norm * np.sum(np.log(ns - ms))


@njit(cache=True)
def dE_fold(fold_norm, m_old, n_old, m_new, n_new):
    return fold_norm * (np.log(n_new - m_new) - np.log(n_old - m_old))


@njit(cache=True)
def fold_energy(p, ms, ns):
    """Folding energy of both LEF families."""
    E = E_fold(ms[:p.N_lef], ns[:p.N_lef], p.fold_norm)
    if p.N_lef2 > 0:
        E += E_fold(ms[p.N_lef:], ns[p.N_lef:], p.fold_norm2)
    return E


# ---- Extra binding energy from BigWig tracks (e.g. RNAPII, other ChIP-seq) ----

@njit(cache=True)
def E_bw(r, BWs, ms, ns):
    """Sum over tracks of r_t * (signal at both legs) / total signal."""
    E = 0.0
    for t in range(BWs.shape[0]):
        E += r[t] * np.sum(BWs[t, ms] + BWs[t, ns]) / np.sum(BWs[t])
    return E


@njit(cache=True)
def dE_bw(r, BWs, m_old, n_old, m_new, n_new):
    dE = 0.0
    for t in range(BWs.shape[0]):
        dE += r[t] * (BWs[t, m_new] + BWs[t, n_new] - BWs[t, m_old] - BWs[t, n_old]) / np.sum(BWs[t])
    return dE


# ---- Epigenetic-model energy: states S_i on beads, coupling J, external field h ----

@njit(cache=True)
def E_epi(S, J, h, epi_norm):
    """-sum_ij J_ij [S_i == S_j] - sum_i h_i S_i  (J = couplings, h = external field)."""
    n, E = len(S), 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0 and S[i] == S[j]:
                E -= J[i, j]
        E -= h[i] * S[i]
    return epi_norm * E


@njit(cache=True)
def dE_epi(S, J, h, epi_norm, k, s_new):
    """Change of E_epi when bead k switches to state s_new."""
    s_old = S[k]
    if s_old == s_new:
        return 0.0
    dE = 0.0
    for j in range(len(S)):
        if j != k and J[k, j] != 0.0:
            if S[j] == s_old: dE += J[k, j]
            if S[j] == s_new: dE -= J[k, j]
    dE += h[k] * (s_old - s_new)
    return epi_norm * dE


# Dynamic coupling: J_ij = 1 on the backbone (|i-j| = 1) and on current LEF loops (m, n), else 0.
# Duplicates count once (J is binary); the diagonal is zero.

@njit(cache=True)
def _first_edge(ms, ns, k):
    """True if LEF k is the first one on its (m, n) pair."""
    for j in range(k):
        if ms[j] == ms[k] and ns[j] == ns[k]:
            return False
    return True


@njit(cache=True)
def _edge_elsewhere(ms, ns, m, n, skip):
    """Does a LEF other than `skip` sit on (m, n), or is (m, n) a backbone bond?"""
    if n == m + 1:
        return True
    for j in range(len(ms)):
        if j != skip and ms[j] == m and ns[j] == n:
            return True
    return False


@njit(cache=True)
def E_epi_dyn(S, ms, ns, h, epi_norm):
    """Same epigenetic energy with J built from the backbone and the current LEF positions."""
    E = 0.0
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            E -= 1.0
    for k in range(len(ms)):
        if ns[k] > ms[k] + 1 and S[ms[k]] == S[ns[k]] and _first_edge(ms, ns, k):
            E -= 1.0
    for i in range(len(S)):
        E -= h[i] * S[i]
    return epi_norm * E


@njit(cache=True)
def dE_epi_dyn(S, ms, ns, h, epi_norm, k, s_new):
    """Change of the dynamic epigenetic energy when bead k switches to s_new."""
    s_old = S[k]
    if s_old == s_new:
        return 0.0
    dE = 0.0
    for q in (k - 1, k + 1):                       # backbone neighbours
        if 0 <= q < len(S):
            if S[q] == s_old: dE += 1.0
            if S[q] == s_new: dE -= 1.0
    for j in range(len(ms)):                       # LEF partners (each pair once)
        if ms[j] == k:
            q = ns[j]
        elif ns[j] == k:
            q = ms[j]
        else:
            continue
        if abs(q - k) == 1 or not _first_edge(ms, ns, j):
            continue
        if S[q] == s_old: dE += 1.0
        if S[q] == s_new: dE -= 1.0
    dE += h[k] * (s_old - s_new)
    return epi_norm * dE


@njit(cache=True)
def dE_epi_move(S, ms, ns, epi_norm, idx, m_new, n_new):
    """Epigenetic energy change when LEF idx moves: its old coupling switches off, the new one on."""
    dE = 0.0
    if not _edge_elsewhere(ms, ns, ms[idx], ns[idx], idx) and S[ms[idx]] == S[ns[idx]]:
        dE += 1.0
    if not _edge_elsewhere(ms, ns, m_new, n_new, idx) and S[m_new] == S[n_new]:
        dE -= 1.0
    return epi_norm * dE


# ---- Total energy and single-LEF energy changes, from the parameter bundle ----

@njit(cache=True)
def total_energy(p, ms, ns, S):
    """E = data + crossing + folding (+ tracks) (+ epigenetic)."""
    E = E_data(p.loss_mode, p.L, p.R, p.J_loss, ms, ns, p.bind_norm)
    E += E_cross(ms, ns, p.k_norm, p.N_lef, p.cross_loop, p.between)
    E += fold_energy(p, ms, ns)
    if p.P.shape[0] > 0:
        E += E_bind_inner(p.P, p.cov_norm, ms, ns)
    if p.BWs.shape[0] > 0:
        E += E_bw(p.r, p.BWs, ms, ns)
    if p.epi_norm != 0.0:
        E += E_epi_dyn(S, ms, ns, p.h, p.epi_norm) if p.epi_dynamic else E_epi(S, p.J, p.h, p.epi_norm)
    return E


@njit(cache=True)
def dE_edge(p, ms, ns, S, m_new, n_new, idx):
    """Energy change if LEF `idx` moves to (m_new, n_new)."""
    norm = p.fold_norm if idx < p.N_lef else p.fold_norm2
    dE = dE_fold(norm, ms[idx], ns[idx], m_new, n_new)
    dE += dE_data(p.loss_mode, p.L, p.R, p.J_loss, p.bind_norm, ms[idx], ns[idx], m_new, n_new)
    dE += dE_cross(ms, ns, m_new, n_new, idx, p.k_norm, p.N_lef, p.cross_loop, p.between)
    if p.P.shape[0] > 0:
        dE += dE_bind_inner(p.P, p.cov_norm, ms[idx], ns[idx], m_new, n_new)
    if p.BWs.shape[0] > 0:
        dE += dE_bw(p.r, p.BWs, ms[idx], ns[idx], m_new, n_new)
    if p.epi_norm != 0.0 and p.epi_dynamic:
        dE += dE_epi_move(S, ms, ns, p.epi_norm, idx, m_new, n_new)
    return dE
