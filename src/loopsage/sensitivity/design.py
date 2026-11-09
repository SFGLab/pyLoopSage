"""Which parameter points to simulate: grid, one-at-a-time, random (quasi-random) or sobol (Saltelli) designs."""
import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class Design:
    kind: str
    axes: list
    points: list           # dicts: id, values {axis name: value}, group (axis varied / role), row

    @property
    def n_points(self) -> int:
        return len(self.points)


def _baseline(spec, ax):
    """Baseline value of an axis (one-at-a-time design): the value in the file if it is on the axis, else the middle one."""
    vals = ax.grid()
    given = spec.system.get(ax.name[7:]) if ax.name.startswith("system.") else spec.base.get(ax.name)
    if given is not None and ax.numeric and not ax.is_range:
        return min(vals, key=lambda v: abs(v - given))
    if given in vals:
        return given
    return vals[len(vals) // 2]


def _sobol_unit(d, n, seed):
    """n x d quasi-random points in [0, 1); n is rounded up to a power of two."""
    from scipy.stats import qmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(max(n, 2)))))


def build_design(spec) -> Design:
    """Points of the scan; each carries `values` for every axis."""
    axes, k = spec.axes, len(spec.axes)
    pts = []
    if spec.design == "grid":
        for i, combo in enumerate(product(*[a.grid() for a in axes])):
            pts.append(dict(id=i, values=dict(zip([a.name for a in axes], combo)), group="grid", row=i))
    elif spec.design == "oat":
        base = {a.name: _baseline(spec, a) for a in axes}
        pts.append(dict(id=0, values=dict(base), group="baseline", row=0))
        for a in axes:
            for v in a.grid():
                if v != base[a.name]:
                    pts.append(dict(id=len(pts), values={**base, a.name: v}, group=a.name, row=len(pts)))
    elif spec.design == "random":
        U = _sobol_unit(k, spec.samples, spec.seed)
        for i, u in enumerate(U):
            pts.append(dict(id=i, values={a.name: a.from_unit(x) for a, x in zip(axes, u)}, group="random", row=i))
    else:                                                                    # sobol: A, B and k matrices AB_i (Saltelli)
        U = _sobol_unit(2 * k, spec.samples, spec.seed)
        A, B = U[:, :k], U[:, k:]
        mats = [("A", A), ("B", B)] + [(f"AB{i}", np.where(np.arange(k) == i, B, A)) for i in range(k)]
        for role, M in mats:
            for j, u in enumerate(M):
                pts.append(dict(id=len(pts), values={a.name: a.from_unit(x) for a, x in zip(axes, u)}, group=role, row=j))
    if len(pts) * spec.replicates > spec.max_runs:
        raise ValueError(f"{len(pts)} points x {spec.replicates} replicates = {len(pts) * spec.replicates} runs; "
                         f"the limit is max_runs = {spec.max_runs} (raise it in [scan], or use design = oat / random / sobol)")
    return Design(spec.design, axes, pts)
