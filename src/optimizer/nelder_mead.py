"""Nelder-Mead search over (K, ln f, ln b, ln T scale) with a final check of the best candidates on longer runs.

K is the number of LEFs of the first family only (n_lef2 is not fitted). T scale multiplies the temperature schedule.
The loss is a noisy Monte Carlo estimate, so the search is only a proposal: the best few candidates (and the
starting point) are re-scored on longer runs with new seeds and the winner is returned.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

LF, LB, LT = (np.log(0.05), np.log(3.0)), (np.log(0.25), np.log(3.0)), (np.log(1.0), np.log(3.0))   # bounds of ln f, ln b, ln T scale


@dataclass
class OptResult:
    start: dict                  # starting parameters and their long-run loss (and its three parts)
    best: dict                   # chosen parameters, long-run loss and parts
    trace: list = field(default_factory=list)        # one [seconds, loss, K, f, b, T scale] per search evaluation
    seconds: float = 0.0
    evaluations: int = 0
    bounds: dict = field(default_factory=dict)       # search range of K, f, b and T scale
    converged: bool = False      # simplex shrank below the tolerance before the evaluation budget ran out
    stopped_early: bool = False  # the loss stopped improving (early stopping) before the budget ran out


def _clip(x, k_bounds):
    """(K, f, b, T scale) from the search vector, inside the bounds."""
    cap = lambda v, lim: float(np.exp(np.clip(v, *lim)))
    return int(np.clip(round(x[0]), *k_bounds)), cap(x[1], LF), cap(x[2], LB), cap(x[3], LT)


class _Stop(Exception):
    """Raised inside the search to stop it early."""


def optimize(objective, K0: int, f0: float, b0: float, n_beads: int, max_evals: int = 60, n_check: int = 3, log=None, tick=None,
             patience: int = 0, min_gain: float = 0.01) -> OptResult:
    """Search from (K0, f0, b0, T scale 1); returns the parameters that scored best on the longer check runs.

    Early stopping: with patience > 0 the search ends when the best loss has not dropped by more than min_gain
    during the last `patience` evaluations (the loss is a noisy estimate, so tiny gains do not count).
    """
    kb = (max(2, K0 // 4), max(min(n_beads // 3, 4 * K0), K0))
    t0, trace = time.time(), []
    mark = [np.inf, 0]                                             # best loss that counted as progress, evaluation of that progress

    def fun(x):
        c = _clip(x, kb)
        loss = objective(*c)
        trace.append([time.time() - t0, loss, *c])
        if log:
            log(len(trace), max_evals, loss, min(r[1] for r in trace))
        if tick:
            tick(loss=loss, best=min(r[1] for r in trace))
        if loss < mark[0] - min_gain:
            mark[:] = [loss, len(trace)]
        elif patience and len(trace) - mark[1] >= patience:
            raise _Stop
        return loss

    x0 = np.array([K0, np.log(f0), np.log(b0), 0.0], float)
    simplex = [x0] + [x0 + d for d in ([max(K0 * 0.3, 2), 0, 0, 0], [0, -1.2, 0, 0], [0, 0, -1.2, 0], [0, 0, 0, 0.5])]
    stopped = False
    try:
        res = minimize(fun, x0, method="Nelder-Mead", options=dict(maxfev=max_evals, xatol=0.02, fatol=0.005, initial_simplex=simplex))
    except _Stop:
        res, stopped = None, True
    seconds = time.time() - t0

    cand, seen = [], set()
    for row in sorted(trace, key=lambda r: r[1]):                  # best distinct candidates + the start
        key = tuple(row[2:])
        if key not in seen and len(cand) < n_check:
            seen.add(key)
            cand.append(key)
    start = (K0, f0, b0, 1.0)
    checked = []
    for c in cand + [start]:                                       # longer runs, new seeds
        loss = objective(*c, longer=3)
        checked.append((loss, c, list(objective.parts)))
        if tick:
            tick(loss=loss, best=min(r[1] for r in trace + [[0, loss]]))
    pack = lambda l, c, parts: dict(K=int(c[0]), f=float(c[1]), b=float(c[2]), T=float(c[3]), loss=float(l), parts=[float(v) for v in parts])
    l_s, c_s, p_s = [r for r in checked if r[1] == start][0]
    l_b, c_b, p_b = min(checked, key=lambda r: r[0])
    return OptResult(pack(l_s, c_s, p_s), pack(l_b, c_b, p_b), trace, seconds, len(trace), dict(K=list(kb), f=[float(np.exp(v)) for v in LF], b=[float(np.exp(v)) for v in LB], T=[float(np.exp(v)) for v in LT]), bool(res is not None and res.success), stopped)
