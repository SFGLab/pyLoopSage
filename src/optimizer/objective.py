"""Loss of one parameter set (K, f, b): how well the MC loop map reproduces the experimental loop map.

loss = w_eig L_eig + 0.5 decay mismatch + w_ins L_ins + 1.0 loop-size shortfall, each term 0 for a perfect match (loss_weights):
- L_eig (Potts model on: (1 - r)/2, r = Pearson correlation of the average Potts state with the A/B reference, B negative): 1 - mean cosine between the absolute values of the leading eigenvectors (N_EIG, paired by eigenvalue) of the correlation matrix of
  log(O/E + 0.2) (O/E = each diagonal divided by its mean, so the decay drops out); 0 = same profiles, 1 = no overlap;
- decay mismatch: variance of log P(s) differences (shape of the decay, scale free);
- L_ins: mean absolute difference of the insulation profiles, each scaled to [0, 1] (0 = identical, 1 = opposite);
- loop-size shortfall: log2(mean input loop / mean simulated loop) when the simulated loops are shorter (see size_shortfall).
The weights w_eig and w_ins depend on the size of the region: for a TAD-scale region the insulation dominates, for a region that spans
compartments the eigenvectors do, and in between they are equal (see loss_weights).
"""
import copy

import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh

from analysis.maps import denoise_map, fill_masked, insulation_score, loop_map, loop_map_from_counts, insulation_window, minmax, ps_curve, state_profile
from monte_carlo.simulation import run_mc
from preproc.readers import read_region_table
from validation.compare import loop_matrix


N_EIG = 3                           # leading eigenvectors compared
SPAN_TAD, SPAN_COMPARTMENT = 1e6, 2e7   # bp: at or below the first the insulation dominates, at or above the second the eigenvectors do
COMMUNICATION = dict(depth=6, hop=0.6)   # longer, stronger chains of communicating loops (target and simulated map alike): long-range structure shows up in sparse loops


def loss_weights(span_bp: float) -> tuple:
    """(eigenvectors, decay, insulation, loop size) weights for a region of `span_bp` bp.

    x runs from 0 (region of SPAN_TAD or less) to 1 (SPAN_COMPARTMENT or more) on a log scale; the eigenvector weight grows
    from 0.5 to 2, the insulation weight falls from 2 to 0.5, and both are 1 half way (about 4.5 Mb).
    """
    x = float(np.clip(np.log(max(span_bp, 1.0) / SPAN_TAD) / np.log(SPAN_COMPARTMENT / SPAN_TAD), 0.0, 1.0))
    return 0.5 * 4 ** x, 0.5, 2.0 * 4 ** -x, 1.0


def weighted(parts, weights) -> float:
    """Weighted loss from the terms (loss_parts, plus the loop-size shortfall when present)."""
    return float(np.dot(weights[:len(parts)], parts))


def target_map(cfg, inputs, geom, path, hic=None, smooth=True) -> np.ndarray:
    """Target contact map (NaN = masked beads).

    From Hi-C: the map denoised over `hic_denoise` beads. From loops: the loops drawn exactly like the simulated ones (dots, domains,
    communication between loops, over longer chains than in the MC map to reveal long-range patterns) and lightly smoothed over `target_smooth` beads, so that the sparse loops look like a Hi-C map.
    """
    if hic is not None:
        return denoise_map(hic, cfg.hic_denoise if smooth else 0)
    c = loop_matrix(read_region_table(path, geom, paired=True), geom)
    k_ref = max(inputs.loops.n_sites // 2, 1)
    m = loop_map_from_counts(c * (2 * k_ref / max(c.sum(), 1e-12)), geom.n_beads, **COMMUNICATION)
    return denoise_map(m, cfg.target_smooth if smooth else 0)


def size_shortfall(mc, loops) -> float:
    """0 when the simulated loops are at least as long on average as the input loops; log2 of the ratio of the mean lengths when shorter.

    Sparse loop data hardly constrain the loop size, so the search can drift to microscopic loops; this one-sided
    term keeps the simulated loops from being much shorter than the measured ones (it never pushes them longer).
    """
    ref = np.asarray(getattr(loops, "loop_lengths", []), float)
    if ref.size == 0:
        return 0.0
    ms, ns = mc.production(mc.Ms), mc.production(mc.Ns)
    sim = np.abs(ns - ms).ravel()
    sim = sim[sim > 0]
    return float(max(0.0, np.log2(ref.mean() / max(sim.mean(), 1.0)))) if sim.size else 0.0


def _oe(m, sep):
    ps = ps_curve(m)
    return m / np.where(ps[sep] > 0, ps[sep], 1.0)


def leading_vectors(oe, k: int = N_EIG) -> np.ndarray:
    """First k eigenvectors (columns) of the correlation matrix of log(O/E + 0.2)."""
    c = np.nan_to_num(np.corrcoef(np.log(oe + 0.2)))
    k = min(k, len(c) - 2)
    if len(c) <= 300:
        return eigh(c, subset_by_index=[len(c) - k, len(c) - 1])[1]
    return eigsh(c, k=k, which="LA", v0=np.ones(len(c)))[1]


def loss_parts(m, t, window: int = 5, state=None, ref=None, smooth=None):
    """(L_eig, decay mismatch, L_ins) of a simulated map `m` against the target `t`.

    With the Potts model on, `state` (average state per bead) and `ref` (A/B reference, B = negative) replace the eigenvector
    comparison: L_eig = (1 - r) / 2 with r their Pearson correlation.
    `smooth` = (insulation, decay) widths in beads: the target's profiles are smoothed that much (sparse loop targets), so longer loops are not punished.
    """
    n = len(m)
    up = np.triu_indices(n, 1)
    ok = np.isfinite(t).sum(0) > n / 2                                 # masked Hi-C beads do not count
    t = fill_masked(t)
    m, t = m / m[up].mean(), t / t[up].mean()
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    if state is not None and ref is not None:
        r = np.corrcoef(state[ok], ref[ok])[0, 1] if state[ok].std() > 1e-12 and ref[ok].std() > 1e-12 else 0.0
        l_eig = (1 - float(r)) / 2
    else:
        u_m, u_t = (leading_vectors(_oe(x, sep)[np.ix_(ok, ok)]) for x in (m, t))
        k = min(u_m.shape[1], u_t.shape[1])
        a_m, a_t = (np.abs(u[:, -k:]) for u in (u_m, u_t))             # absolute values: the sign of an eigenvector is arbitrary
        l_eig = 1 - float(np.sum(a_m * a_t / np.linalg.norm(a_m, axis=0) / np.linalg.norm(a_t, axis=0))) / k   # mean cosine, 0..1
    half = slice(1, n // 2)
    ps_t = ps_curve(t) if smooth is None else gaussian_filter1d(ps_curve(t), smooth[1], mode="nearest")
    d = np.log(ps_curve(m)[half] + 0.02) - np.log(ps_t[half] + 0.02)
    ins_m, ins_t = (insulation_score(x, window)[window:-window] for x in (m, t))
    a, b = minmax(ins_m), minmax(ins_t if smooth is None else gaussian_filter1d(ins_t, smooth[0], mode="nearest"))
    return max(l_eig, 0.0), float(np.var(d)), float(np.abs(a - b).mean())


class Objective:
    """Runs a short MC for given (K, f, b, T scale) and scores its loop map; each call uses a new seed.

    Only the first LEF family (n_lef) is fitted; n_lef2 and folding_coeff2 stay as configured.
    T scale multiplies the whole temperature schedule (t_init and t_final), so annealing keeps its shape.
    """

    def __init__(self, cfg, inputs, geom, target, steps: int, seed: int, ref=None, sparse: bool = False):
        self.cfg, self.inputs, self.geom, self.target = cfg, inputs, geom, target
        self.ref = ref if float(cfg.epi_coeff) > 0 else None             # Potts on: the average state is compared with this A/B reference
        self.steps, self.seed, self.calls, self.parts = int(steps), int(seed), 0, None
        self.weights = loss_weights(geom.length_bp)
        w = insulation_window(geom)
        self.smooth = (float(w), max(2.0, w / 3)) if sparse else None      # sparse loop target: strongly smoothed insulation and decay profiles

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
        state = None if self.ref is None else state_profile(mc, c.n_epi_states)
        self.parts = (*loss_parts(loop_map(mc, self.geom.n_beads, **COMMUNICATION), self.target, insulation_window(self.geom), state, self.ref, self.smooth), size_shortfall(mc, self.inputs.loops))
        return weighted(self.parts, self.weights)
