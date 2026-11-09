"""Loss of one parameter set (K, f, b): how well the MC loop map reproduces the experimental loop map.

loss = w_eig L_eig + 0.5 decay mismatch + w_ins L_ins + 1.0 loop-size mismatch, each term 0 for a perfect match (loss_weights):
- L_eig (Potts model on: (1 - r)/2, r = Pearson correlation of the average Potts state with the A/B reference, B negative): 1 - mean cosine between the absolute values of the leading eigenvectors (N_EIG, paired by eigenvalue) of the correlation matrix of
  log(O/E + 0.2) (O/E = each diagonal divided by its mean, so the decay drops out); 0 = same profiles, 1 = no overlap;
- decay mismatch: variance of log P(s) differences (shape of the decay, scale free);
- L_ins: mean absolute difference of the insulation profiles, each scaled to [0, 1] (0 = identical, 1 = opposite); for a loop target the profile is the loop coverage
  (every loop counts on all beads between its anchors, for the input and the simulated loops alike);
- loop-size mismatch: |log2(median simulated loop / median input loop)| beyond a factor of 2 (see size_mismatch); it stops the search from shrinking the loops to nothing and from stacking them over the whole region.
The weights w_eig and w_ins depend on the size of the region: for a TAD-scale region the insulation dominates, for a region that spans
compartments the eigenvectors do, and in between they are equal (see loss_weights).
"""
import copy

import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh

from loopsage.analysis.maps import denoise_map, fill_masked, insulation_score, loop_counts, loop_coverage, loop_decay, loop_map_from_counts, insulation_window, minmax, ps_curve, state_profile
from loopsage.monte_carlo.simulation import run_mc
from loopsage.preproc.readers import read_region_table
from loopsage.validation.compare import loop_matrix


N_EIG = 3                           # leading eigenvectors compared
SPAN_MID, SPAN_SLOPE = 3e7, 3.0     # bp, steepness: the eigenvector weight turns on around 3 A/B blocks (~10 Mb) and dominates from ~30 Mb
COMMUNICATION = dict(depth=6, hop=0.6)   # longer, stronger chains of communicating loops (target and simulated map alike): long-range structure shows up in sparse loops


def loss_weights(span_bp: float) -> tuple:
    """(eigenvectors, decay, insulation, loop size) weights for a region of `span_bp` bp.

    A logistic switch in ln(span): s = 1 / (1 + (SPAN_MID / span)^SPAN_SLOPE); w_eig = 2 s, w_ins = max(2 (1 - s), 0.25).
    Biology: a TAD is ~0.1-1 Mb, an A/B block a few Mb. Up to ~5 Mb PC1 is nearly meaningless (weight ~ 0), it starts to count once a
    region holds 2-3 blocks (~10 Mb) and dominates once it holds many (>= ~30 Mb, up to a whole chromosome); both weights are 1 at SPAN_MID.
    """
    sw = 1.0 / (1.0 + (SPAN_MID / max(span_bp, 1.0)) ** SPAN_SLOPE)
    return 2.0 * sw, 0.5, max(2.0 * (1.0 - sw), 0.25), 1.0


def weighted(parts, weights) -> float:
    """Weighted loss from the terms (loss_parts, plus the loop-size mismatch when present)."""
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


def loop_profiles(c) -> tuple:
    """(coverage, decay) of a loop-count matrix: both count every loop over all the beads / separations it spans."""
    return loop_coverage(c), loop_decay(c)


def target_coverage(geom, path) -> tuple:
    """(coverage, decay) of the input loops: the insulation and decay profiles of a loop target."""
    return loop_profiles(loop_matrix(read_region_table(path, geom, paired=True), geom))


def size_mismatch(mc, loops, tol: float = 1.0) -> float:
    """How far the median simulated loop length is from the median input loop length, in octaves beyond a tolerance (0 inside a factor 2^tol).

    Sparse loop data hardly constrain the loop size: the search can drift to microscopic loops, or (with the
    folding reward at its limit) to loops stacked over the whole region. The median is used because the input lengths
    have a heavy tail, so their mean would ask for far more nesting than the extruders can supply.
    """
    ref = np.asarray(getattr(loops, "loop_lengths", []), float)
    if ref.size == 0:
        return 0.0
    ms, ns = mc.production(mc.Ms), mc.production(mc.Ns)
    sim = np.abs(ns - ms).ravel()
    sim = sim[sim > 0]
    if not sim.size:
        return 0.0
    return float(max(0.0, abs(np.log2(max(np.median(sim), 1.0) / max(np.median(ref), 1.0))) - tol))


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


def state_loss(state, ref, ok=None) -> float:
    """(1 - r) / 2, r = Pearson correlation of the average Potts state with the A/B reference (0 = same pattern, 1 = mirror image)."""
    ok = np.isfinite(ref) if ok is None else ok & np.isfinite(ref)
    if state[ok].std() < 1e-12 or ref[ok].std() < 1e-12:
        return 0.5
    return (1 - float(np.corrcoef(state[ok], ref[ok])[0, 1])) / 2


def short_cfg(cfg, steps: int, seed: int):
    """Copy of the configuration for a short MC run (one replica)."""
    c = copy.copy(cfg)
    c.n_steps, c.mc_step, c.burnin, c.replicas, c.seed = steps, max(steps // 200, 1), steps // 4, 1, seed
    return c


def loss_parts(m, t, window: int = 5, state=None, ref=None, smooth=None, cover=None, eig: bool = True, track: float = 0.0):
    """(L_eig, decay mismatch, L_ins) of a simulated map `m` against the target `t`.

    With the Potts model on, `state` (average state per bead) and `ref` (A/B reference, B = negative) replace the eigenvector
    comparison: L_eig = (1 - r) / 2 with r their Pearson correlation.
    `eig` = False skips the eigenvector term (0): the loop-extrusion stage of a Potts run fits only insulation and decay.
    `track` = smoothing (beads) of the insulation profiles of `m` and `t`.
    `cover` = (simulated, input) (coverage, decay) of the loops: the insulation and decay profiles when the target is a set of loops (the whole area between anchors counts).
    `smooth` = (insulation, decay) widths in beads: the target's profiles are smoothed that much (sparse loop targets), so longer loops are not punished.
    """
    n = len(m)
    up = np.triu_indices(n, 1)
    ok = np.isfinite(t).sum(0) > n / 2                                 # masked Hi-C beads do not count
    t = fill_masked(t)
    m, t = m / m[up].mean(), t / t[up].mean()
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    if not eig:
        l_eig = 0.0
    elif state is not None and ref is not None:
        l_eig = state_loss(state, ref, ok)
    else:
        u_m, u_t = (leading_vectors(_oe(x, sep)[np.ix_(ok, ok)]) for x in (m, t))
        k = min(u_m.shape[1], u_t.shape[1])
        a_m, a_t = (np.abs(u[:, -k:]) for u in (u_m, u_t))             # absolute values: the sign of an eigenvector is arbitrary
        l_eig = 1 - float(np.sum(a_m * a_t / np.linalg.norm(a_m, axis=0) / np.linalg.norm(a_t, axis=0))) / k   # mean cosine, 0..1
    half = slice(1, n // 2)
    if cover is None:
        (ps_m, ps_t), (ins_m, ins_t) = (ps_curve(m), ps_curve(t)), (insulation_score(x, window, track)[window:-window] for x in (m, t))
    else:
        (cov_m, ps_m), (cov_t, ps_t) = cover
        ps_m, ps_t = (np.asarray(p, float) / max(np.mean(p[half]), 1e-12) for p in (ps_m, ps_t))     # decays scaled to unit mean
        ins_m, ins_t = (np.asarray(x, float)[window:-window] for x in (cov_m, cov_t))
    ps_t = ps_t if smooth is None else gaussian_filter1d(ps_t, smooth[1], mode="nearest")
    d = np.log(ps_m[half] + 0.02) - np.log(ps_t[half] + 0.02)
    a, b = minmax(ins_m), minmax(ins_t if smooth is None else gaussian_filter1d(ins_t, smooth[0], mode="nearest"))
    return max(l_eig, 0.0), float(np.var(d)), float(np.abs(a - b).mean())


class Objective:
    """Runs a short MC for given (K, f, b, T scale) and scores its loop map; each call uses a new seed.

    Only the first LEF family (n_lef) is fitted; n_lef2 and folding_coeff2 stay as configured.
    T scale multiplies the whole temperature schedule (t_init and t_final), so annealing keeps its shape.
    """

    def __init__(self, cfg, inputs, geom, target, steps: int, seed: int, ref=None, sparse: bool = False, cover=None, loops_only: bool = False):
        self.cfg, self.inputs, self.geom, self.target = cfg, inputs, geom, target
        self.loops_only = loops_only                                     # stage 1 of a Potts run: Potts off, only insulation, decay and loop size are fitted
        if loops_only:
            self.cfg = copy.copy(cfg)
            self.cfg.epi_coeff, ref = 0.0, None
        self.ref = ref if float(cfg.epi_coeff) > 0 else None             # Potts on: the average state is compared with this A/B reference
        self.cover = cover                                               # input loop coverage (loop targets)
        self.steps, self.seed, self.calls, self.parts = int(steps), int(seed), 0, None
        self.weights = loss_weights(geom.length_bp)
        if loops_only:
            self.weights = (0.0, *self.weights[1:])
        w = insulation_window(geom)
        self.smooth = (float(w), max(2.0, w / 3)) if sparse else None      # sparse loop target: strongly smoothed insulation and decay profiles

    def __call__(self, K, f, b, T=1.0, longer: int = 1) -> float:
        """Loss of (K, f, b, T scale); longer > 1 uses a run that many times longer (final check). Loss parts in .parts."""
        c = short_cfg(self.cfg, self.steps * longer, self.seed + 7919 * self.calls)
        c.n_lef, c.folding_coeff, c.bind_coeff = int(K), float(f), float(b)
        c.t_init, c.t_final = self.cfg.t_init * float(T), self.cfg.t_final * float(T)
        self.calls += 1
        mc = run_mc(c, self.inputs, self.geom, quiet=True)
        state = None if self.ref is None else state_profile(mc, c.n_epi_states)
        n = self.geom.n_beads
        cnt = loop_counts(mc, n)
        cover = None if self.cover is None else (loop_profiles(cnt), self.cover)
        self.parts = (*loss_parts(loop_map_from_counts(cnt, n, **COMMUNICATION), self.target, insulation_window(self.geom), state, self.ref, self.smooth, cover, not self.loops_only, float(self.cfg.track_smooth)), size_mismatch(mc, self.inputs.loops))
        return weighted(self.parts, self.weights)


class PottsObjective(Objective):
    """Stage 2 (Potts model on): the full weighted loss for given (epi_coeff, p_epigenetic), the loop-extrusion parameters of `cfg` being fixed.

    PC1 (average Potts state vs the A/B reference) enters with its region-size weight, so it is balanced against the insulation, decay and loop size:
    the coupling is only pushed as far as the compartments matter for this region, and not at the cost of the TADs.
    """

    def __call__(self, epi, p, longer: int = 1) -> float:
        self.cfg.epi_coeff, self.cfg.p_epigenetic = float(epi), float(p)
        return super().__call__(self.cfg.n_lef, self.cfg.folding_coeff, self.cfg.bind_coeff, 1.0, longer)
