"""Benchmark: infer (K, f, b) of the loop-extrusion energy for an artificial 1000-bead polymer (Hi-C, CTCF, anchors).

Boltzmann-based methods use the equilibrium law P(config) ~ exp(-E/T): the ideal-gas closed form (A) and
importance reweighting of one MC ensemble (C). They are compared with black-box MC search and with the defaults.
Nothing in src/ is used or modified.   Run: python demo/optimizer/bench1000.py
"""
import json
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent
N, T, W = 1000, 1.0, 10                              # beads, temperature, insulation window (beads)
TRUTH = dict(K=55, f=0.1, b=0.3)                     # parameters that generated the artificial Hi-C
BOUNDS = dict(K=(8, 140), lf=(np.log(0.01), np.log(3)), lb=(np.log(0.005), np.log(3)))
EVAL, FINAL = (100_000, 30_000, 20), (500_000, 60_000, 20)        # (sweeps, burn-in, thinning) of cheap / final runs


# ---- Artificial data: CTCF sites, anchor profiles, Hi-C --------------------------------------------------------

def contrast(v):
    """Mean-scale, z-score, tanh (same input processing as LoopSage)."""
    v = v / v.mean()
    return np.tanh((v - v.mean()) / (v.std() + 1e-6))


def profiles(pos, orient, strength):
    """Left/right anchor profiles L, R from forward / reverse CTCF sites (peaks smoothed over ~1 bead)."""
    L, R = np.zeros(N), np.zeros(N)
    np.add.at(L, pos[orient > 0], strength[orient > 0])
    np.add.at(R, pos[orient < 0], strength[orient < 0])
    return contrast(gaussian_filter(L, 1.0)), contrast(gaussian_filter(R, 1.0))


def make_sites(rng, n_sites=110):
    """Artificial CTCF sites: spaced positions, random orientation (+1 forward / -1 reverse), lognormal strength."""
    pos = np.sort(rng.choice(np.arange(5, N - 5, 4), n_sites, replace=False))
    return pos, rng.choice([-1, 1], n_sites), rng.lognormal(0, 0.5, n_sites)


def observed_sites(rng, pos, orient, strength):
    """What the analyst sees: 20% of sites missed, 10% spurious, noisy strengths."""
    keep = rng.random(len(pos)) > 0.2
    extra = rng.choice(np.arange(5, N - 5), int(0.1 * len(pos)), replace=False)
    return (np.r_[pos[keep], extra], np.r_[orient[keep], rng.choice([-1, 1], len(extra))],
            np.r_[strength[keep] * rng.lognormal(0, 0.2, keep.sum()), rng.lognormal(-0.5, 0.3, len(extra))])


# ---- Exact Boltzmann sampler of loops (symmetric proposals, hard exclusion of crossings / shared anchors) --------

@njit(cache=True)
def _chain(K, fa, bb, L, R, n_sweeps, burn, thin, seed, Ms, Ns, rho, Fs, Bs):
    np.random.seed(seed)
    n = L.shape[0]
    ms, ns = np.zeros(K, np.int64), np.zeros(K, np.int64)
    gap = (n - 4) // K
    for k in range(K):
        ms[k] = 1 + k * gap
        ns[k] = ms[k] + 2
    s = 0
    for sweep in range(n_sweeps):
        for _ in range(K):
            j = np.random.randint(0, K)
            u = np.random.rand()
            if u < 0.45:                                       # shift each leg independently
                w = 1 if np.random.rand() < 0.34 else (4 if np.random.rand() < 0.5 else 16)
                m, e = ms[j] + np.random.randint(-w, w + 1), ns[j] + np.random.randint(-w, w + 1)
            elif u < 0.9:                                      # slide the whole loop
                w = 1 if np.random.rand() < 0.34 else (8 if np.random.rand() < 0.5 else 64)
                d = np.random.randint(-w, w + 1)
                m, e = ms[j] + d, ns[j] + d
            else:                                              # rebind uniformly
                m, e = np.random.randint(0, n), np.random.randint(0, n)
            if m < 0 or e >= n or e - m < 2:
                continue
            bad = False
            for i in range(K):
                if i != j:
                    a, c = ms[i], ns[i]
                    if (m < a < e < c) or (a < m < c < e) or m == a or m == c or e == a or e == c:
                        bad = True
                        break
            if bad:
                continue
            dE = fa * (np.log(e - m) - np.log(ns[j] - ms[j])) + bb * (L[m] + R[e] - L[ms[j]] - R[ns[j]])
            if dE <= 0 or np.exp(-dE / T) > np.random.rand():
                ms[j], ns[j] = m, e
        if sweep >= burn and (sweep - burn) % thin == 0 and s < Ms.shape[1]:
            f_, b_ = 0.0, 0.0
            for k in range(K):
                Ms[k, s], Ns[k, s] = ms[k], ns[k]
                rho[ms[k], ns[k]] += 1.0
                f_ += np.log(ns[k] - ms[k])
                b_ += L[ms[k]] + R[ns[k]]
            Fs[s], Bs[s] = f_, b_
            s += 1
    return s


def norms(K, n_sites):
    """Energy normalisations of the folding and binding terms (as in LoopSage)."""
    return -N / (K * np.log(N / K)), -2.0 * N / n_sites


class Model:
    """Energy E = f F0 sum ln(l) + b B0 sum (L_m + R_n) for given anchor profiles."""

    def __init__(self, L, R, n_sites):
        self.L, self.R, self.n_sites = L, R, n_sites

    def sample(self, K, f, b, budget, seed):
        """Snapshots, density rho(m, n) and per-snapshot statistics from the exact sampler."""
        sweeps, burn, thin = budget
        F0, B0 = norms(K, self.n_sites)
        S = (sweeps - burn) // thin
        Ms, Ns, rho = np.zeros((K, S), np.int64), np.zeros((K, S), np.int64), np.zeros((N, N))
        Fs, Bs = np.zeros(S), np.zeros(S)
        _chain(K, f * F0, b * B0, self.L, self.R, sweeps, burn, thin, seed, Ms, Ns, rho, Fs, Bs)
        return dict(Ms=Ms, Ns=Ns, rho=rho / S, F=Fs, B=Bs, K=K, f=f, b=b)

    def ideal_gas(self, K, f, b):
        """Closed-form density of K independent loops, rho ~ K exp(-eps/T) / Z (Boltzmann, no exclusion)."""
        F0, B0 = norms(K, self.n_sites)
        ell = np.arange(N)[None, :] - np.arange(N)[:, None]
        eps = f * F0 * np.log(np.where(ell >= 2, ell, 1)) + b * B0 * (self.L[:, None] + self.R[None, :])
        w = np.where(ell >= 2, np.exp(-(eps - eps[ell >= 2].min()) / T), 0.0)
        return K * w / w.sum()

    def reweight(self, ens, f, b):
        """Density at (f, b) from an ensemble drawn at (ens.f, ens.b), same K (importance weights exp(-dE/T)); also ESS fraction."""
        F0, B0 = norms(ens["K"], self.n_sites)
        dE = (f - ens["f"]) * F0 * ens["F"] + (b - ens["b"]) * B0 * ens["B"]
        w = np.exp(-(dE - dE.min()) / T)
        w /= w.sum()
        rho = np.zeros((N, N))
        np.add.at(rho, (ens["Ms"].ravel(), ens["Ns"].ravel()), np.tile(w, ens["Ms"].shape[0]))
        return rho * 1.0, 1.0 / (np.sum(w ** 2) * len(w))


# ---- Maps and scores -----------------------------------------------------------------------------------------

_I = np.arange(N)
_SEP = np.abs(_I[:, None] - _I[None, :])
_BG = 1.0 / (1.0 + _SEP) ** 0.8
_UP = np.triu_indices(N, 2)


def to_map(rho, bg=0.2):
    """Contact map: number of loops spanning both beads (2D prefix sums), slightly blurred, plus a decay background."""
    S = np.cumsum(rho, axis=0)[:, ::-1].cumsum(axis=1)[:, ::-1]
    M = S[np.minimum(_I[:, None], _I[None, :]), np.maximum(_I[:, None], _I[None, :])]
    M = gaussian_filter(M, 1.0)
    return M + bg * M[_UP].mean() * _BG / _BG[_UP].mean()


def noisy_hic(M, rng, depth=30.0):
    """Artificial Hi-C: Poisson counts of the (symmetric) map at the given mean depth."""
    C = rng.poisson(depth * M / M[_UP].mean()).astype(float)
    C = np.triu(C) + np.triu(C, 1).T
    return gaussian_filter(C, 1.0)                              # light smoothing, as for binned Hi-C


def features(M):
    """P(s) = mean contact at separation s, and the insulation score (diamond block, window W)."""
    ps = np.array([np.diagonal(M, s).mean() for s in range(1, N)])
    P = np.pad(M.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    i = np.arange(W, N - W)
    blk = lambda r0, r1, c0, c1: P[r1, c1] - P[r0, c1] - P[r1, c0] + P[r0, c0]
    ins = blk(i - W, i, i + 1, i + 1 + W) / W ** 2
    return ps, ins


_BAND = (_SEP >= 2) & (_SEP <= 300) & (_I[:, None] < _I[None, :])


def obs_exp(M):
    """Observed / expected: each diagonal divided by its mean, so that only structure beyond the decay remains."""
    ps = np.r_[1.0, [np.diagonal(M, s).mean() for s in range(1, N)]]
    return M / ps[_SEP]


def oe_corr(M, Mt):
    """Pearson correlation of log(O/E + 0.2) over the pixels 2 <= |i - j| <= 300 (decay removed)."""
    a, b = np.log(obs_exp(M)[_BAND] + 0.2), np.log(obs_exp(Mt)[_BAND] + 0.2)
    return float(np.corrcoef(a, b)[0, 1])


def parts(M, Mt):
    """(1 - Pearson of O/E maps, decay mismatch, 1 - Pearson of insulation): 0 = perfect."""
    M, Mt = M / M[_UP].mean(), Mt / Mt[_UP].mean()
    r_map = oe_corr(M, Mt)
    (p, i), (pt, it) = features(M), features(Mt)
    d = np.log(p[1:N // 2] + 0.02) - np.log(pt[1:N // 2] + 0.02)
    return 1 - r_map, float(np.mean((d - d.mean()) ** 2)), 1 - np.corrcoef(i, it)[0, 1]


def metrics(M, Mt):
    """Final scores: insulation r, P(s) r (log), pixel r (log raw and O/E), mean per-diagonal r over s = 2..300, and the loss."""
    M, Mt = M / M[_UP].mean(), Mt / Mt[_UP].mean()
    (p, i), (pt, it) = features(M), features(Mt)
    diag = np.mean([np.corrcoef(np.diagonal(M, s), np.diagonal(Mt, s))[0, 1] for s in range(2, 301)])
    lm, lp, li = parts(M, Mt)
    return dict(insulation_r=float(1 - li), decay_r=float(np.corrcoef(np.log(p[1:N // 2]), np.log(pt[1:N // 2]))[0, 1]),
                pixel_r=float(np.corrcoef(np.log(M[_UP] + 0.05), np.log(Mt[_UP] + 0.05))[0, 1]), oe_r=float(1 - lm), diagonal_r=float(diag), decay_mismatch=lp, loss=lm + lp + li)


# ---- Optimisers ----------------------------------------------------------------------------------------------

class Run:
    """Wall-clock trace of (time, loss, best loss) over the evaluations of one optimiser."""

    def __init__(self):
        self.t0, self.trace, self.best = time.time(), [], np.inf

    def log(self, loss):
        self.best = min(self.best, loss)
        self.trace.append((time.time() - self.t0, loss, self.best))
        return loss


def clip(K, lf, lb):
    return int(np.clip(round(K), *BOUNDS["K"])), float(np.clip(lf, *BOUNDS["lf"])), float(np.clip(lb, *BOUNDS["lb"]))


def search(model, hic, method, default, seed=0, n_rand=40, maxfev=60):
    """Return fitted (K, f, b), trace and number of evaluations for one method."""
    run, rng, nev = Run(), np.random.default_rng(seed), [0]
    K0 = default["K"]

    def mc_loss(K, lf, lb):
        nev[0] += 1
        e = model.sample(K, np.exp(lf), np.exp(lb), EVAL, seed + nev[0])
        return run.log(sum(parts(to_map(e["rho"]), hic)))

    if method == "random":
        cand = [(K0, 0.0, 0.0)] + [(rng.integers(*BOUNDS["K"]), *rng.uniform(BOUNDS["lf"][0], BOUNDS["lf"][1], 1), *rng.uniform(BOUNDS["lb"][0], BOUNDS["lb"][1], 1)) for _ in range(n_rand)]
        best = min(cand, key=lambda c: mc_loss(*clip(*c)))
        return clip(*best), run
    if method == "neldermead":
        r = minimize(lambda x: mc_loss(*clip(*x)), [K0, 0.0, 0.0], method="Nelder-Mead", options=dict(maxfev=maxfev, xatol=0.02, fatol=0.01, initial_simplex=[[K0, 0, 0], [K0 + 15, 0, 0], [K0, -1.5, 0], [K0, 0, -1.5]]))
        return clip(*r.x), run
    if method == "ideal":                                       # A: closed-form Boltzmann density, no MC in the loop
        best = (np.inf, None)
        for K in range(15, 125, 15):
            f_ = lambda x: run.log(sum(parts(to_map(model.ideal_gas(K, *np.exp(clip(K, *x)[1:]))), hic)))
            r = minimize(f_, [0.0, -1.0], method="Nelder-Mead", options=dict(maxfev=30, xatol=0.05, fatol=0.01))
            if r.fun < best[0]:
                best = (r.fun, clip(K, *r.x))
        return best[1], run
    if method == "ideal+reweight":                              # A then C: MC ensembles reweighted over (f, b)
        K, lf, lb = search(model, hic, "ideal", default, seed)[0]
        for cycle in range(3):
            ens = model.sample(K, np.exp(lf), np.exp(lb), EVAL, seed + 100 + cycle)

            def fun(x):
                rho, ess = model.reweight(ens, *np.exp(clip(K, *x)[1:]))
                return run.log(sum(parts(to_map(rho), hic)) + (50 * (0.3 - ess) if ess < 0.3 else 0.0))
            r = minimize(fun, [lf, lb], method="Nelder-Mead", options=dict(maxfev=40, xatol=0.02, fatol=0.005))
            _, lf, lb = clip(K, *r.x)
        return (K, lf, lb), run
    raise ValueError(method)


# ---- Experiment ----------------------------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(1)
    pos, ori, st = make_sites(rng)
    true_m = Model(*profiles(pos, ori, st), len(pos))
    truth = true_m.sample(TRUTH["K"], TRUTH["f"], TRUTH["b"], FINAL, 123)
    hic = noisy_hic(to_map(truth["rho"]), rng)                  # artificial Hi-C
    op, oo, os_ = observed_sites(rng, pos, ori, st)
    model = Model(*profiles(op, oo, os_), len(op))              # the analyst's (imperfect) CTCF annotation
    default = dict(K=len(op) // 2, f=1.0, b=1.0)                # LoopSage defaults: K = N_anchors / 2, f = b = 1
    print(f"{len(pos)} true / {len(op)} observed CTCF sites; default K = {default['K']}", flush=True)

    fits, traces = {}, {}
    fits["default parameters"] = ((default["K"], 0.0, 0.0), None)
    for label, method in (("random search (MC)", "random"), ("Nelder-Mead (MC)", "neldermead"),
                          ("A: ideal gas (Boltzmann)", "ideal"), ("A+C: ideal gas + reweighting", "ideal+reweight")):
        t0 = time.time()
        fits[label] = search(model, hic, method, default)
        print(f"{label}: {time.time() - t0:.0f} s, K={fits[label][0][0]}, f={np.exp(fits[label][0][1]):.3g}, b={np.exp(fits[label][0][2]):.3g}", flush=True)

    res = {}
    oracle = true_m.sample(TRUTH["K"], TRUTH["f"], TRUTH["b"], FINAL, 777)
    res["oracle (true parameters and sites)"] = dict(K=TRUTH["K"], f=TRUTH["f"], b=TRUTH["b"], **metrics(to_map(oracle["rho"]), hic))
    maps = {"Hi-C": hic}
    for label, ((K, lf, lb), run) in fits.items():
        e = model.sample(K, np.exp(lf), np.exp(lb), FINAL, 999)
        M = to_map(e["rho"])
        maps[label] = M
        r = dict(K=K, f=float(np.exp(lf)), b=float(np.exp(lb)), **metrics(M, hic))
        if run is not None:
            tr = np.array(run.trace)
            r.update(seconds=float(tr[-1, 0]), evaluations=len(tr))
            traces[label] = tr[:, [0, 2]].tolist()
        res[label] = r
    base = res["default parameters"]["loss"]
    for label, r in res.items():
        r["loss_reduction_pct"] = 100 * (1 - r["loss"] / base)
        if label in traces:                                      # time to reach 95% of the method's own improvement
            tr = np.array(traces[label])
            goal = base - 0.95 * (base - r["loss"])
            hit = np.flatnonzero(tr[:, 1] <= goal)
            r["seconds_to_95pct"] = float(tr[hit[0], 0]) if len(hit) else r["seconds"]
    json.dump(dict(results=res, traces=traces, truth=TRUTH, n_true_sites=len(pos), n_obs_sites=len(op)), open(OUT / "results.json", "w"), indent=1)
    np.savez_compressed(OUT / "maps.npz", **{k.replace(" ", "_"): v for k, v in maps.items()})
    print(json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
