"""Run a scan: build the sandbox of each point, sample it in memory, keep only the metrics. Parallel over runs, with a cache."""
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from loopsage.sensitivity import metrics as M
from loopsage.sensitivity.design import Design, build_design
from loopsage.sensitivity.spec import ALIASES, ScanSpec

CODE_VERSION = "2"


# ---- The synthetic system ------------------------------------------------------------------------

def make_system(sys: dict):
    """Sandbox for the [system] settings: CTCF sites (even or random layout, chosen orientations) and optional compartments."""
    from loopsage.sandbox import Sandbox
    n, bp, nc = int(sys["n_beads"]), int(sys["bp_per_bead"]), int(sys["n_ctcf"])
    if int(sys.get("ctcf_spacing", 0)) > 0:
        nc = max(n // int(sys["ctcf_spacing"]), 2)
    if not 2 <= nc <= n // 3:
        raise ValueError(f"n_ctcf must be between 2 and n_beads/3 (got {nc} for {n} beads)")
    rng = np.random.default_rng(int(sys["seed"]))
    if sys["layout"] == "even":
        beads = np.unique(np.round(np.linspace(0.06, 0.94, nc) * (n - 1)).astype(int))
    elif sys["layout"] == "random":
        beads = []
        for b in rng.permutation(np.arange(3, n - 3)):
            if all(abs(b - x) >= 3 for x in beads):
                beads.append(int(b))
            if len(beads) == nc:
                break
        beads = np.sort(beads)
    else:
        raise ValueError("layout must be 'even' or 'random'")
    orient = {"convergent": ["forward", "reverse"] * nc, "forward": ["forward"] * nc, "reverse": ["reverse"] * nc,
              "random": list(rng.choice(["forward", "reverse"], nc))}.get(sys["orientation"])
    if orient is None:
        raise ValueError("orientation must be convergent, random, forward or reverse")
    strength = float(sys["strength"]) * np.exp(float(sys["strength_spread"]) * rng.normal(size=len(beads)))
    box = Sandbox(f"chr1:0-{n * bp}", n_beads=n)
    box.add_ctcfs([int(b) * bp + bp // 2 for b in beads], orient[:len(beads)], list(strength))
    if sys["compartments"] == "random":
        box.random_compartments(mean_block_mb=float(sys["comp_block_mb"]), seed=int(sys["seed"]))
    anchor, r = np.zeros(n, bool), int(sys["bound_radius"])
    for b in beads:
        anchor[max(b - r, 0):b + r + 1] = True
    return box, anchor


def resolve(spec_base: dict, values: dict):
    """(config parameters, system settings) for one point."""
    params, sysv = dict(spec_base), {}
    for name, v in values.items():
        if name.startswith("system."):
            sysv[name[7:]] = v
        else:
            for key in ALIASES.get(name, (name,)):
                params[key] = v
    return params, sysv


def make_config(box, params, task):
    """Config of a run: scan defaults for snapshots and burn-in, one thread, one chain, quiet."""
    p = dict(params)
    n = int(p["n_steps"])
    p.setdefault("mc_step", max(n // int(task["snapshots"]), 1))
    p.setdefault("burnin", int(n * float(task["burnin_fraction"])))
    p.update(seed=int(task["seed"]), quiet=True, threads=1, replicas=1)
    if task["md"]:
        p.update(run_md=True, sim_step=int(task["md_steps"]), platform=task["md_platform"], md_skip_burnin=True)
        if task["topology"]:
            p.update(detect_knots=True, track_crossings=True, knots_stride=1)
    return box.config(**p)


# ---- One run ------------------------------------------------------------------------------------------

def _load_metrics_file(path):
    if path:
        spec = importlib.util.spec_from_file_location("loopsage_scan_metrics", path)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _thin(mc, frames):
    """A copy of the chain with `frames` evenly spaced production snapshots (so the 3D step stays cheap)."""
    from dataclasses import replace
    T, b = mc.n_samples, mc.burnin_idx
    idx = np.unique(np.linspace(b, T - 1, frames).astype(int))
    cut = lambda a: a[..., idx]
    return replace(mc, Ms=cut(mc.Ms), Ns=cut(mc.Ns), Es=cut(mc.Es), Ks=cut(mc.Ks), Fs=cut(mc.Fs), Bs=cut(mc.Bs), ufs=cut(mc.ufs),
                   Ts=cut(mc.Ts), epi=cut(mc.epi), meta=dict(mc.meta, burnin=0), replicas=[])


def run_task(task: dict) -> dict:
    """Simulate one point and one replicate; returns a small record (series, summaries, profiles)."""
    import loopsage as ls
    t0, rec = time.time(), dict(key=task["key"], point=task["point"], rep=task["rep"], seed=task["seed"], ok=False, error="")
    try:
        _load_metrics_file(task.get("metrics_file", ""))
        params, sysv = resolve(task["base"], task["values"])
        box, anchor = make_system({**task["system"], **sysv})
        cfg = make_config(box, params, task)
        geom = ls.geometry(cfg)
        mc = ls.run_mc(cfg, ls.read_inputs(cfg, geom, **box.arrays()), geom)
        ctx = M.context(mc, geom, anchor)
        names = [m for m in task["metrics"] if m in M.METRICS]
        series = {m: M.METRICS[m].fn(ctx).astype(np.float32) for m in names}
        b = mc.burnin_idx
        rec.update(series=series, steps=ls.results.steps(mc), burnin_idx=b, n_lef=mc.meta["n_lef"], n_sites=mc.meta["n_sites"],
                   summary={m: float(v[b:].mean()) for m, v in series.items()}, tau={m: M.tau_int(v[b:]) for m, v in series.items()},
                   first_half={m: float(v[b:b + (len(v) - b) // 2].mean()) for m, v in series.items()},
                   second_half={m: float(v[b + (len(v) - b) // 2:].mean()) for m, v in series.items()},
                   stats={m: dict(std=float(v[b:].std()), bim=M.bimodality(v[b:]), u4=M.binder(v[b:])) for m, v in series.items()},
                   samples={m: v[b:][np.linspace(0, len(v) - b - 1, min(40, len(v) - b)).astype(int)] for m, v in series.items()},
                   profiles=M.profiles(ctx, b), n_beads=geom.n_beads, bp=geom.bp_per_bead, acc=mc.acc.tolist())
        if task["md"]:
            sub = _thin(mc, int(task["md_frames"]))
            md = ls.run_md(cfg, sub, geom)
            rec["md"] = M.md_metrics(md, anchor)
            if task["topology"]:
                rec["md"].update(M.topology_metrics(md, sub, cfg))
        rec["ok"] = True
    except Exception as e:                                                     # a failed point must not stop the scan
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["runtime_s"] = time.time() - t0
    return rec


def _init_worker():
    """One thread per worker process: the scan parallelises over runs."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = "1"


# ---- Tasks, cache, scheduling -----------------------------------------------------------------------

def make_tasks(spec: ScanSpec, design: Design) -> list:
    """Every (point, replicate) as a plain dict; replicate r uses seed + r at every point (common random numbers)."""
    names = M.select(spec.metrics)
    tasks = []
    for pt in design.points:
        for r in range(spec.replicates):
            t = dict(point=pt["id"], rep=r, seed=spec.seed + r, values=pt["values"], base=spec.base, system=spec.system, metrics=names,
                     snapshots=spec.snapshots, burnin_fraction=spec.burnin_fraction, md=spec.md, topology=spec.topology, md_frames=spec.md_frames,
                     md_steps=spec.md_steps, md_platform=spec.md_platform, metrics_file=spec.metrics_file)
            ident = {k: t[k] for k in ("seed", "base", "system", "metrics", "snapshots", "burnin_fraction", "md", "topology", "md_frames", "md_steps")}
            ident["values"] = pt["values"]
            ident["v"] = CODE_VERSION
            t["key"] = hashlib.sha1(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:20]
            tasks.append(t)
    return tasks


def cost(t) -> float:
    """Relative cost of a run (Monte Carlo ~ steps x LEFs^1.2; the 3D frames dominate when they are on)."""
    params, sysv = resolve(t["base"], t["values"])
    nc = sysv.get("n_ctcf", t["system"]["n_ctcf"])
    n_lef = params.get("n_lef") or max(nc // 2, 1)
    c = float(params["n_steps"]) * n_lef ** 1.2
    return c + (2e6 * t["md_frames"] * (sysv.get("n_beads", t["system"]["n_beads"]) / 300) ** 2 if t["md"] else 0.0)


class Cache:
    """Records by key in one pickle file next to the report (no per-run output)."""

    def __init__(self, path, on):
        self.path, self.on, self.data, self.hits = Path(path), on, {}, 0
        if on and self.path.is_file():
            try:
                self.data = pickle.loads(self.path.read_bytes())
            except Exception:
                self.data = {}

    def save(self):
        if self.on:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(self.data, protocol=4))
            tmp.replace(self.path)


@dataclass
class ScanResult:
    spec: ScanSpec
    design: Design
    tasks: list
    records: dict                    # key -> record
    wall_s: float = 0.0
    cache_hits: int = 0
    workers: int = 1
    notes: list = field(default_factory=list)

    @property
    def table(self) -> pd.DataFrame:
        """One row per run: axis values, replicate, the production mean of every metric (`<name>`), its autocorrelation time
        (`tau:<name>`) and the 3D metrics when they were run."""
        rows = []
        for t in self.tasks:
            r = self.records.get(t["key"])
            if not r or not r["ok"]:
                continue
            row = dict(point=t["point"], rep=t["rep"], **{a.label: t["values"][a.name] for a in self.design.axes})
            row.update(r["summary"])
            row.update({f"tau:{k}": v for k, v in r["tau"].items()})
            for k, st in r["stats"].items():
                row.update({f"std:{k}": st["std"], f"bim:{k}": st["bim"], f"u4:{k}": st["u4"]})
            row.update(r.get("md", {}))
            row.update(runtime_s=r["runtime_s"], n_lef=r["n_lef"])
            rows.append(row)
        return pd.DataFrame(rows)

    def report(self, path) -> Path:
        """Write the combined report (sensitivity_report.html) and return its path."""
        from loopsage.sensitivity.report import write_report
        return write_report(self, path)

    @property
    def failures(self) -> list:
        return [(t["point"], t["rep"], self.records[t["key"]]["error"]) for t in self.tasks if t["key"] in self.records and not self.records[t["key"]]["ok"]]


def estimate(spec: ScanSpec, workers: int = None):
    """Time one typical run and extrapolate: (seconds, number of runs, time of the reference run)."""
    design = build_design(spec)
    tasks = make_tasks(spec, design)
    ref = sorted(tasks, key=cost)[len(tasks) // 2]
    t0 = time.time()
    rec = run_task(ref)
    if not rec["ok"]:
        raise RuntimeError("the test run failed: " + rec["error"])
    one = time.time() - t0
    w = workers or spec.workers or (os.cpu_count() or 1)
    total = sum(cost(t) for t in tasks) / cost(ref) * rec["runtime_s"] / max(min(w, len(tasks)), 1)
    return total, len(tasks), one


def run_scan(spec: ScanSpec, out, workers: int = None, progress=True, replot_only: bool = False) -> ScanResult:
    """Run (or load from the cache) every point of the scan. `out` is the folder of the report; the cache file lives there."""
    from loopsage.common.runtime import stage_bar
    design = build_design(spec)
    tasks = make_tasks(spec, design)
    cache = Cache(Path(out) / ".scan_cache.pkl", spec.cache)
    todo = [t for t in tasks if t["key"] not in cache.data]
    if replot_only and todo:
        raise ValueError(f"--replot needs a finished scan: {len(todo)} of {len(tasks)} runs are not in the cache")
    cache.hits = len(tasks) - len({t["key"] for t in todo}) if tasks else 0
    todo = list({t["key"]: t for t in todo}.values())                              # identical runs once
    todo.sort(key=cost, reverse=True)                                              # longest first: the cores finish together
    w = max(1, min(workers or spec.workers or (os.cpu_count() or 1), len(todo) or 1))
    t0 = time.time()
    if todo:
        _execute(todo, cache, w, progress, stage_bar)
    cache.save()
    return ScanResult(spec, design, tasks, {**cache.data} if spec.cache else {t["key"]: cache.data[t["key"]] for t in tasks}, time.time() - t0, cache.hits, w)


def _context():
    """Fork on Linux (scripts need no __main__ guard); spawn elsewhere."""
    import sys
    return get_context("fork" if sys.platform.startswith("linux") else "spawn")


def _execute(todo, cache, w, progress, stage_bar):
    from contextlib import ExitStack, nullcontext
    done = 0
    with ExitStack() as stack:
        ex = stack.enter_context(ProcessPoolExecutor(w, mp_context=_context(), initializer=_init_worker)) if w > 1 else None
        advance = stack.enter_context(stage_bar(f"Scan: {len(todo)} runs on {w} worker(s)", len(todo)) if progress else nullcontext(lambda n=1: None))
        if ex is None:
            for t in todo:
                cache.data[t["key"]] = run_task(t)
                advance(); done += 1
                if done % 25 == 0:
                    cache.save()
            return
        futs = {ex.submit(run_task, t): t for t in todo}
        try:
            for f in as_completed(futs):
                cache.data[futs[f]["key"]] = f.result()
                advance(); done += 1
                if done % 25 == 0:
                    cache.save()
        except BaseException:
            for f in futs:
                f.cancel()
            cache.save()
            raise
