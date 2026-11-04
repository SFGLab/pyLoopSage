"""Loss of one parameter set (K, f, b): how well the MC loop map reproduces the experimental loop map.

loss = (1 - r_OE) + decay mismatch + (1 - r_ins), each term 0 for a perfect match:
- r_OE: Pearson r of log(O/E + 0.2) (observed / expected = each diagonal divided by its mean, so the decay drops out);
- decay mismatch: variance of log P(s) differences (shape of the decay, scale free);
- r_ins: Pearson r of the insulation scores.
"""
import copy

import numpy as np

from analysis.maps import insulation_score, loop_map, loop_map_from_counts, ps_curve
from monte_carlo.simulation import run_mc
from preproc.readers import read_region_table
from validation.compare import loop_matrix


def target_map(cfg, inputs, geom, path) -> np.ndarray:
    """Experimental loop map: the input loops drawn like the simulated ones, scaled to the default number of LEFs."""
    c = loop_matrix(read_region_table(path, geom, paired=True), geom)
    k_ref = max(inputs.loops.n_sites // 2, 1)
    return loop_map_from_counts(c * (2 * k_ref / max(c.sum(), 1e-12)), geom.n_beads)


def _oe(m, sep):
    ps = ps_curve(m)
    return m / np.where(ps[sep] > 0, ps[sep], 1.0)


def loss_parts(m, t, window: int = 5):
    """(1 - r_OE, decay mismatch, 1 - r_ins) of a simulated map `m` against the target `t`."""
    n = len(m)
    up = np.triu_indices(n, 1)
    m, t = m / m[up].mean(), t / t[up].mean()
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    band = (sep >= 2) & (sep <= max(n // 3, 5)) & (np.triu(np.ones((n, n)), 1) > 0)
    r_map = np.corrcoef(np.log(_oe(m, sep)[band] + 0.2), np.log(_oe(t, sep)[band] + 0.2))[0, 1]
    half = slice(1, n // 2)
    d = np.log(ps_curve(m)[half] + 0.02) - np.log(ps_curve(t)[half] + 0.02)
    a, b = insulation_score(m, window)[window:-window], insulation_score(t, window)[window:-window]
    r_ins = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
    return 1 - r_map, float(np.var(d)), 1 - r_ins


class Objective:
    """Runs a short MC for given (K, f, b, T scale) and scores its loop map; each call uses a new seed.

    Only the first LEF family (n_lef) is fitted; n_lef2 and folding_coeff2 stay as configured.
    T scale multiplies the whole temperature schedule (t_init and t_final), so annealing keeps its shape.
    """

    def __init__(self, cfg, inputs, geom, target, steps: int, seed: int):
        self.cfg, self.inputs, self.geom, self.target = cfg, inputs, geom, target
        self.steps, self.seed, self.calls, self.parts = int(steps), int(seed), 0, None

    def __call__(self, K, f, b, T=1.0, longer: int = 1) -> float:
        """Loss of (K, f, b, T scale); longer > 1 uses a run that many times longer (final check). Loss parts in .parts."""
        c = copy.copy(self.cfg)
        steps = self.steps * longer
        c.n_lef, c.folding_coeff, c.bind_coeff = int(K), float(f), float(b)
        c.t_init, c.t_final = self.cfg.t_init * float(T), self.cfg.t_final * float(T)
        c.n_steps, c.mc_step, c.burnin, c.replicas = steps, max(steps // 200, 1), steps // 4, 1
        c.seed = self.seed + 7919 * self.calls
        self.calls += 1
        mc = run_mc(c, self.inputs, self.geom, quiet=True)
        self.parts = loss_parts(loop_map(mc, self.geom.n_beads), self.target)
        return float(sum(self.parts))
