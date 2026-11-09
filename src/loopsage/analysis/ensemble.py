"""Loop statistics of the equilibrium ensemble.

- Convergence and equilibrium diagnostics of the Monte Carlo run.
- Extra insight metrics for the loop ensemble: anchor recovery, LEF dynamics, length agreement.
- Agreement between independent MC chains (replicas).
- Epigenetic model: average state, same-state map, agreement of loop ends, energy over time.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp, spearmanr


# ---- Loop statistics of the equilibrium ensemble ---------------------------

_MOVES = ("Rebinding", "Sliding", "Epigenetic-state change")


def loop_lengths(res) -> np.ndarray:
    """All loop lengths (beads) of every LEF in every post-burn-in snapshot."""
    return (res.production(res.Ns) - res.production(res.Ms)).ravel()


def occupancy(res, n_beads: int) -> np.ndarray:
    """Average number of LEF legs sitting on each bead (anchor probability)."""
    legs = np.concatenate([res.production(res.Ms).ravel(), res.production(res.Ns).ravel()])
    return np.bincount(legs, minlength=n_beads) / res.production(res.Es).shape[0]


def acceptance_table(res) -> list:
    """[(move, proposed, accepted, rate)] for each move type that was used."""
    return [(name, int(p), int(a), a / p) for name, (p, a) in zip(_MOVES, res.acc) if p > 0]


def loop_pairs_of_frame(ms, ns) -> np.ndarray:
    """Unique loops (left, right) of one MC snapshot as an (k, 2) int array; zero-length loops dropped."""
    lo, hi = np.minimum(ms, ns), np.maximum(ms, ns)
    pairs = np.unique(np.c_[lo, hi][hi > lo], axis=0)
    return pairs.astype(int)


# ---- Convergence and equilibrium diagnostics of the Monte Carlo run --------

TERMS = {"Es": "Total energy", "Bs": "Binding energy", "Ks": "Crossing energy",
         "Fs": "Folding energy", "ufs": "Unfolded fraction"}


def acf(x, max_lag=None) -> np.ndarray:
    """Normalised autocorrelation function (FFT based), lag 0 = 1."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    a = a / a[0] if a[0] > 0 else np.zeros(n)
    return a[: (max_lag or n // 2) + 1]


def integrated_time(x) -> float:
    """Integrated autocorrelation time in samples (Sokal self-consistent window)."""
    rho = acf(x, len(x) - 1)
    tau = 1.0
    for m in range(1, len(rho)):
        tau = 1 + 2 * rho[1:m + 1].sum()
        if m >= 5 * tau:
            break
    return float(max(tau, 1.0))


def detrend(x, deg: int = 3) -> np.ndarray:
    """Remove a polynomial trend (annealing drifts the energy)."""
    t = np.arange(len(x))
    return np.asarray(x) - np.polyval(np.polyfit(t, x, deg), t)


def equilibrium_summary(res) -> dict:
    """Post-burn-in mean/std per term, drift between halves, and effective sample size."""
    out, annealing = {}, res.meta["method"] == "Annealing"
    for key in TERMS:
        x = res.production(getattr(res, key))
        half = len(x) // 2
        drift = (x[half:].mean() - x[:half].mean()) / (x.std() + 1e-12) if half > 1 else 0.0
        out[key] = dict(mean=float(x.mean()), std=float(x.std()), drift=float(drift))
    x = res.production(res.Fs)
    tau = integrated_time(detrend(x) if annealing else x) if len(x) > 10 else 1.0
    out["tau"], out["ess"], out["n"] = tau, len(x) / tau, len(x)
    return out


# ---- Extra insight metrics for the loop ensemble: anchor recovery, LEF dynamics, length agreement ----

def anchor_profile(inputs) -> np.ndarray:
    """Input anchor strength per bead (L + R)."""
    return inputs.loops.L + inputs.loops.R


def anchor_recovery(res, inputs, sigma: float = 1.0) -> dict:
    """How well LEF legs sit on the strongest input anchors.

    Anchors = beads above the 75th percentile of the non-zero input signal.
    recall = anchors that hold a LEF leg; precision = occupied beads that are anchors.
    """
    occ, sig = occupancy(res, len(inputs.loops.L)), anchor_profile(inputs)
    nz = sig[sig > 0]
    anchors = sig >= (np.percentile(nz, 75) if nz.size else np.inf)
    used = occ > 0.5 * occ.max() if occ.max() > 0 else occ > 0
    rho = spearmanr(gaussian_filter1d(occ, sigma), gaussian_filter1d(sig, sigma))[0] if sig.std() > 0 else np.nan
    return dict(n_anchor_beads=int(anchors.sum()), n_used_beads=int(used.sum()),
                recall=float((anchors & used).sum() / max(anchors.sum(), 1)),
                precision=float((anchors & used).sum() / max(used.sum(), 1)), spearman=float(rho))


def lef_dynamics(res, jump_beads: int = 20) -> dict:
    """Stability of loops between consecutive post-burn-in snapshots."""
    ms, ns = res.production(res.Ms), res.production(res.Ns)
    same = (ms[:, 1:] == ms[:, :-1]) & (ns[:, 1:] == ns[:, :-1])
    jumps = (np.abs(ms[:, 1:] - ms[:, :-1]) > jump_beads) | (np.abs(ns[:, 1:] - ns[:, :-1]) > jump_beads)
    step, n_int = res.meta["mc_step"], max(same.shape[1], 1)
    changes = (~same).sum(axis=1)
    return dict(stable_fraction=float(same.mean()),
                dwell_steps=float(n_int * step / np.maximum(changes, 1).mean()),
                rebinding_per_1000=float(jumps.sum(axis=1).mean() / (n_int * step) * 1000),
                frozen_lefs=float((changes == 0).mean()))


def length_agreement(res, inputs, geom):
    """Input vs simulated loop lengths in kb: (input_kb, sim_kb, KS statistic, p-value)."""
    inp = geom.beads_to_bp(inputs.loops.loop_lengths) / 1e3
    sim = geom.beads_to_bp(loop_lengths(res)) / 1e3
    if inp.size < 2 or sim.size < 2:
        return inp, sim, np.nan, np.nan
    ks = ks_2samp(inp, sim)
    return inp, sim, float(ks.statistic), float(ks.pvalue)


# ---- Agreement between independent MC chains (replicas) --------------------

def r_hat(chains) -> float:
    """Gelman-Rubin potential scale reduction of a scalar series per chain (1 = chains agree)."""
    n = min(len(c) for c in chains)
    x = np.array([c[-n:] for c in chains], float)
    m = len(x)
    w, b = x.var(axis=1, ddof=1).mean(), n * x.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w)) if w > 0 else float("nan")


def replica_summary(res) -> dict:
    """Per-chain energy statistics, occupancy profiles and their pairwise correlation."""
    chains = res.chains
    prod = [c.production(c.Es) for c in chains]
    occ = np.array([occupancy(c, c.meta["n_beads"]) for c in chains])
    corr = np.corrcoef(occ) if len(chains) > 1 else np.ones((1, 1))
    iu = np.triu_indices(len(chains), 1)
    return dict(energy=prod, mean=np.array([e.mean() for e in prod]), std=np.array([e.std() for e in prod]), occ=occ, corr=corr,
                rhat=r_hat(prod) if len(chains) > 1 else float("nan"), min_corr=float(corr[iu].min()) if len(iu[0]) else float("nan"))


# ---- Epigenetic model: states of beads and their link to the loops ----------

def state_fractions(S, n_states: int) -> np.ndarray:
    """(n_states, n_beads): fraction of snapshots in which each bead is in each state."""
    return np.stack([(S == s).mean(axis=1) for s in range(n_states)])


def same_state_map(S, n_states: int) -> np.ndarray:
    """P(S_i == S_j) over snapshots for every pair of beads (1 on the diagonal)."""
    out = np.zeros((S.shape[0],) * 2)
    for s in range(n_states):
        x = (S == s).astype(float)
        out += x @ x.T
    return out / S.shape[1]


def loop_state_agreement(res, n_states: int, n_bins: int = 10) -> dict:
    """Do the two ends of a loop share an epigenetic state more often than expected at that distance?

    Returns bin centres (beads), observed rate for loop ends and baseline rate for any two beads
    the same distance apart (baseline = mean of the same-state map along the matching diagonal).
    """
    S = res.production(res.epi)
    M = same_state_map(S, n_states)
    m, n = res.production(res.Ms), res.production(res.Ns)              # (n_lef, n_snapshots)
    d = (n - m).ravel()
    same = (np.take_along_axis(S, m, 0) == np.take_along_axis(S, n, 0)).ravel()
    base_d = np.array([np.diagonal(M, k).mean() if k < M.shape[0] else np.nan for k in range(M.shape[0])])
    edges = np.unique(np.geomspace(max(d.min(), 1), d.max() + 1, n_bins + 1).astype(int))
    idx = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    keep = [b for b in range(len(edges) - 1) if (idx == b).any()]
    return dict(
        sep=np.array([d[idx == b].mean() for b in keep]),
        loops=np.array([same[idx == b].mean() for b in keep]),
        baseline=np.array([np.nanmean(base_d[d[idx == b]]) for b in keep]),
        overall=float(same.mean()), overall_base=float(np.nanmean(base_d[d])))


def epigenetic_energy_series(p, res) -> tuple:
    """Epigenetic-model energy at every saved snapshot: (MC steps, energies)."""
    from loopsage.monte_carlo.energy import E_epi, E_epi_dyn
    out = np.empty(res.n_samples)
    for k in range(res.n_samples):
        S = res.epi[:, k].astype(np.int64)
        ms, ns = res.Ms[:, k].astype(np.int64), res.Ns[:, k].astype(np.int64)
        out[k] = E_epi_dyn(S, ms, ns, p.h, p.epi_norm) if p.epi_dynamic else E_epi(S, p.J, p.h, p.epi_norm)
    return np.arange(res.n_samples) * res.meta["mc_step"], out
