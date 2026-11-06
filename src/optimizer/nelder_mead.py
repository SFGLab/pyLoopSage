"""Nelder-Mead search over (K, ln f, ln b, ln T scale) with a final check of the best candidates on longer runs.

With the Potts model on there is a second search, over (ln epi_coeff, p_epigenetic), after the loop-extrusion one (optimize_potts): it keeps the loop-extrusion
parameters and balances PC1 against the TAD terms with the same region-size weights.

K is the number of LEFs of the first family only (n_lef2 is not fitted). T scale multiplies the temperature schedule.
The loss is a noisy Monte Carlo estimate, so the search is only a proposal: the best few candidates (and the
starting point) are re-scored on longer runs with new seeds and the winner is returned.
"""
import select
import sys
import termios
import threading
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
    potts: dict = None           # second stage (Potts model on): start, best, trace, bounds, evaluations, seconds, converged, stopped_early
    converged: bool = False      # simplex shrank below the tolerance before the evaluation budget ran out
    stopped_early: bool = False  # the loss stopped improving (early stopping) before the budget ran out
    by_user: bool = False        # the user pressed Enter and chose to keep the best parameters so far


def _clip(x, k_bounds):
    """(K, f, b, T scale) from the search vector, inside the bounds."""
    cap = lambda v, lim: float(np.exp(np.clip(v, *lim)))
    return int(np.clip(round(x[0]), *k_bounds)), cap(x[1], LF), cap(x[2], LB), cap(x[3], LT)


class _Stop(Exception):
    """Raised inside the search to stop it early."""


_ENTER, _DONE = threading.Event(), threading.Event()       # Enter pressed / search over (stops the watcher)


def _read_enter() -> bool:
    """True if a line was waiting on the terminal (it is read and dropped)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def _watch_enter(notice) -> None:
    """Background thread: sets _ENTER as soon as the user presses Enter, and says so at once (an evaluation can take a while)."""
    while not _DONE.is_set():
        try:
            if _read_enter() and not _ENTER.is_set():
                _ENTER.set()
                notice("Enter pressed: pausing as soon as the current evaluation ends.")
            time.sleep(0.2)
        except Exception:                                          # no usable terminal: the check in enter_pressed still works
            return


def enter_pressed() -> bool:
    """True once if the user pressed Enter since the last call (also checks the terminal itself, in case the watcher thread is not running)."""
    seen = _ENTER.is_set()
    _ENTER.clear()
    try:
        return seen or (sys.stdin.isatty() and _read_enter())
    except Exception:
        return seen


def _drop_typed_input() -> None:
    """Discard keys typed while the search was paused, so they do not answer the next question."""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def ask_keep_best(names, unit: str = ""):
    """Returns ask(trace) for the search: after Enter, shows the best point so far and asks whether to stop and keep it."""
    def ask(trace, prompt=input) -> bool:
        best = min(trace, key=lambda r: r[1])
        pars = ", ".join(f"{k} = {v:.3g}" for k, v in zip(names, best[2:]))
        try:
            ans = prompt(f"\n[optimizer] {len(trace)} evaluations done. Best so far: {pars} (loss {best[1]:.3f}).\n[optimizer] Search paused. Stop and keep these parameters? (Y/n): ").strip().lower()
        except EOFError:
            return False
        _drop_typed_input()
        return not ans.startswith("n")                    # Enter = yes
    return ask


def _search(objective, x0, simplex, decode, start, max_evals, n_check, log, tick, patience, min_gain, ask=None):
    """Nelder-Mead on decode(x) -> parameters, then a check of the best distinct candidates and `start` on 3x longer runs with new seeds.

    Early stopping: with patience > 0 the search ends when the best loss has not dropped by more than min_gain
    during the last `patience` evaluations (the loss is a noisy estimate, so tiny gains do not count).
    `ask(trace)` is offered whenever the user presses Enter; if it returns True the search ends and the best point so far goes to the checks.
    Returns (trace, checked [(loss, parameters, parts)], converged, stopped_early, seconds).
    """
    t0, trace = time.time(), []
    watcher = None
    if ask and sys.stdin.isatty():
        _DONE.clear()
        watcher = threading.Thread(target=_watch_enter, args=(getattr(tick, "notice", print),), daemon=True)
        watcher.start()
    parts_log = []                                                 # loss parts of every evaluation (used when the user stops the search)
    mark = [np.inf, 0]                                             # best loss that counted as progress, evaluation of that progress

    def fun(x):
        c = decode(x)
        loss = objective(*c)
        trace.append([time.time() - t0, loss, *c])
        parts_log.append(list(objective.parts))
        if log:
            log(len(trace), max_evals, loss, min(r[1] for r in trace))
        if tick:
            tick(loss=loss, best=min(r[1] for r in trace))
        if ask and enter_pressed() and ask(trace, getattr(tick, "prompt", input)):
            objective.user_stopped = True
            raise _Stop
        if loss < mark[0] - min_gain:
            mark[:] = [loss, len(trace)]
        elif patience and len(trace) - mark[1] >= patience:
            raise _Stop
        return loss

    stopped = False
    try:
        res = minimize(fun, x0, method="Nelder-Mead", options=dict(maxfev=max_evals, xatol=0.02, fatol=0.005, initial_simplex=simplex))
    except _Stop:
        res, stopped = None, True
    _DONE.set()
    if watcher:
        watcher.join(1.0)
    seconds = time.time() - t0
    if getattr(objective, "user_stopped", False):                  # the user keeps the best point so far: no long checks, straight to the run
        k = min(range(len(trace)), key=lambda i: trace[i][1])
        checked = [(trace[0][1], start, parts_log[0]), (trace[k][1], tuple(trace[k][2:]), parts_log[k])]
        if tick:
            tick(n=len(checked), loss=trace[k][1], best=trace[k][1])
        return trace, checked, False, True, seconds
    cand, seen = [], set()
    for row in sorted(trace, key=lambda r: r[1]):                  # best distinct candidates + the start
        key = tuple(row[2:])
        if key not in seen and len(cand) < n_check:
            seen.add(key)
            cand.append(key)
    checked = []
    for c in cand + [start]:                                       # longer runs, new seeds
        loss = objective(*c, longer=3)
        checked.append((loss, c, list(objective.parts)))
        if tick:
            tick(loss=loss, best=min(r[1] for r in trace + [[0, loss]]))
    return trace, checked, bool(res is not None and res.success), stopped, seconds


def optimize(objective, K0: int, f0: float, b0: float, n_beads: int, max_evals: int = 60, n_check: int = 3, log=None, tick=None,
             patience: int = 0, min_gain: float = 0.01, ask=None) -> OptResult:
    """Search from (K0, f0, b0, T scale 1); returns the parameters that scored best on the longer check runs."""
    kb = (max(2, K0 // 4), max(min(n_beads // 3, 4 * K0), K0))
    x0 = np.array([K0, np.log(f0), np.log(b0), 0.0], float)
    simplex = [x0] + [x0 + d for d in ([max(K0 * 0.3, 2), 0, 0, 0], [0, -1.2, 0, 0], [0, 0, -1.2, 0], [0, 0, 0, 0.5])]
    start = (K0, f0, b0, 1.0)
    trace, checked, conv, stopped, seconds = _search(objective, x0, simplex, lambda x: _clip(x, kb), start, max_evals, n_check, log, tick, patience, min_gain, ask)
    pack = lambda l, c, parts: dict(K=int(c[0]), f=float(c[1]), b=float(c[2]), T=float(c[3]), loss=float(l), parts=[float(v) for v in parts])
    l_s, c_s, p_s = [r for r in checked if r[1] == start][0]
    l_b, c_b, p_b = min(checked, key=lambda r: r[0])
    return OptResult(pack(l_s, c_s, p_s), pack(l_b, c_b, p_b), trace, seconds, len(trace), dict(K=list(kb), f=[float(np.exp(v)) for v in LF], b=[float(np.exp(v)) for v in LB], T=[float(np.exp(v)) for v in LT]), converged=conv, stopped_early=stopped, by_user=getattr(objective, "user_stopped", False))


PE = (0.1, 0.9)             # bounds of p_epigenetic
PSPAN = 10.0                # epi_coeff is searched within [start / PSPAN, start * PSPAN]


def optimize_potts(objective, e0: float, p0: float, max_evals: int = 20, n_check: int = 3, log=None, tick=None, patience: int = 0, min_gain: float = 0.01, ask=None) -> dict:
    """Second stage (the loop-extrusion parameters are fixed): search (epi_coeff, p_epigenetic) from the start, same checks as `optimize`."""
    le = (np.log(e0 / PSPAN), np.log(e0 * PSPAN))
    decode = lambda x: (float(np.exp(np.clip(x[0], *le))), float(np.clip(x[1], *PE)))
    x0 = np.array([np.log(e0), p0], float)
    simplex = [x0, x0 + [1.2, 0], x0 + [0, -0.2 if p0 > 0.6 else 0.2]]
    trace, checked, conv, stopped, seconds = _search(objective, x0, simplex, decode, (e0, p0), max_evals, n_check, log, tick, patience, min_gain, ask)
    pack = lambda l, c, parts: dict(epi=float(c[0]), p=float(c[1]), loss=float(l), parts=[float(v) for v in parts])
    l_s, c_s, p_s = [r for r in checked if r[1] == (e0, p0)][0]
    l_b, c_b, p_b = min(checked, key=lambda r: r[0])
    return dict(start=pack(l_s, c_s, p_s), best=pack(l_b, c_b, p_b), trace=trace, seconds=seconds, evaluations=len(trace), converged=conv, stopped_early=stopped,
                bounds=dict(epi=[float(np.exp(v)) for v in le], p=list(PE)))
