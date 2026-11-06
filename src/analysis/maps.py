"""Matrix helpers for contact maps.

- Hi-C-like contact map from the loop ensemble (no 3D needed), with the optional epigenetic checkerboard.
- Hi-C style features of a contact map: PC1 (compartments), insulation (TADs) and P(s) decay.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.signal import find_peaks


# ---- Matrix helpers for contact maps ---------------------------------------

def minmax(x) -> np.ndarray:
    """Scale to [0, 1]."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def ps_curve(mat) -> np.ndarray:
    """Mean contact vs. bead separation s (diagonal decay, P(s))."""
    return np.array([np.diagonal(mat, k).mean() for k in range(mat.shape[0])])


def fill_masked(mat) -> np.ndarray:
    """Replace NaN (masked beads) by the mean of their diagonal, so they are neutral in O/E, P(s) and insulation."""
    m = np.array(mat, float)
    if np.isfinite(m).all():
        return m
    n = len(m)
    for k in range(n):
        d = np.diagonal(m, k)
        bad = ~np.isfinite(d)
        if bad.any():
            ix = np.arange(n - k)[bad]
            m[ix, ix + k] = m[ix + k, ix] = np.nanmean(d) if (~bad).any() else 0.0
    return m


def denoise_map(mat, sigma: float) -> np.ndarray:
    """Gaussian denoising of the observed/expected pattern, keeping the decay and the masked beads (NaN).

    Each diagonal is divided by its mean, the ratio is blurred over `sigma` beads (masked pairs ignored), and the decay is put back.
    """
    m = np.array(mat, float)
    if sigma <= 0:
        return m
    seen = np.isfinite(m)
    n, f = len(m), fill_masked(m)
    exp = np.array([np.diagonal(f, k).mean() for k in range(n)])
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    e = np.where(exp[sep] > 0, exp[sep], 1.0)
    num, den = gaussian_filter(f / e, sigma), gaussian_filter(np.ones_like(f), sigma)
    out = num / den * e
    out = (out + out.T) / 2
    out[~seen] = np.nan
    return out


def insulation_score(mat, window: int = 5) -> np.ndarray:
    """Diamond insulation score: mean contact across each bead."""
    n = mat.shape[0]
    ins = np.zeros(n)
    for i in range(window, n - window):
        ins[i] = mat[i - window:i, i + 1:i + window + 1].mean()
    return ins


INSULATION_BP = 100_000      # span of the insulation diamond: TAD scale, whatever the bead size


def insulation_window(geom, bp: int = INSULATION_BP) -> int:
    """Diamond half-width in beads for a span of `bp` (at least 5 beads)."""
    return max(5, int(round(bp / geom.bp_per_bead)))


def insulation_profile(mat, window: int = 5) -> np.ndarray:
    """Insulation score with its undefined ends (`window` beads each) filled by the nearest value, so they do not look like data."""
    ins = insulation_score(mat, window)
    return np.pad(ins[window:-window], window, mode="edge") if len(ins) > 2 * window else ins


def pearson_maps(a, b) -> float:
    """Pearson r between the upper triangles (k>=1) of two matrices."""
    iu = np.triu_indices(a.shape[0], k=1)
    return float(np.corrcoef(a[iu], b[iu])[0, 1])


# ---- Hi-C-like contact map from the loop ensemble (no 3D needed) ------------

def loop_counts(res, n_beads: int) -> np.ndarray:
    """Symmetric (N, N) matrix: average number of loops joining beads i and j per snapshot."""
    ms, ns = res.production(res.Ms), res.production(res.Ns)
    lo, hi = np.minimum(ms, ns).ravel(), np.maximum(ms, ns).ravel()
    ok = hi > lo
    a = np.bincount(lo[ok] * n_beads + hi[ok], minlength=n_beads ** 2).reshape(n_beads, n_beads) / ms.shape[1]
    return a + a.T


def loop_map(res, n_beads: int, dot=1.0, domain=0.1, hop=0.4, depth=3, sigma=0.8, base=0.3) -> np.ndarray:
    """Contact map from loops, built from snapshot-averaged loop counts (fast: a few N x N operations).

    - dots: a loop (i, j) gives a smooth peak at (i, j) and (j, i), blurred by `sigma` beads;
    - domain: pairs inside a loop (the triangle between the diagonal and the dot) get a weaker `domain` enrichment;
    - communication: loops (i, j) and (j, k) or (j+1, k) (anchors on the same or adjacent beads) also enrich (i, k), recursively up to `depth` loops,
      each extra hop weaker by the factor `hop`;
    - base: weak 1/(1+s) polymer background (s = bead separation) so that the P(s) decay is defined.
    Stripes are not added by hand; they appear when LEFs hold one anchor and the other leg moves.
    """
    return loop_map_from_counts(loop_counts(res, n_beads), n_beads, dot, domain, hop, depth, sigma, base)


def loop_map_from_counts(c, n_beads: int, dot=1.0, domain=0.1, hop=0.4, depth=3, sigma=0.8, base=0.3) -> np.ndarray:
    """The same map from a symmetric (N, N) loop-count matrix (simulated or experimental)."""
    cs = sparse.csr_matrix(c)                                    # loop matrices are sparse: chains of loops are cheap this way
    near = sparse.identity(n_beads, format="csr") + sparse.eye(n_beads, k=1, format="csr") + sparse.eye(n_beads, k=-1, format="csr")   # anchors on the same or adjacent beads communicate
    link, step = cs.copy(), cs.copy()
    for _ in range(depth - 1):                                   # i-j-k chains of increasing length
        step = (hop * (step @ near @ cs)).tolil()
        step.setdiag(0)
        step = step.tocsr()
        link = link + step
    link = link.toarray()
    link = (link + link.T) / 2                                   # zeroed diagonals break exact symmetry
    up = np.triu(c) - np.diag(np.diag(c))                        # loops once, as (left, right)
    cover = np.cumsum(np.cumsum(up[:, ::-1], axis=1)[:, ::-1], axis=0)      # loops with left <= i and right >= j
    cover = np.triu(cover, 1)
    sep = np.abs(np.subtract.outer(np.arange(n_beads), np.arange(n_beads)))
    return base / (1 + sep) + dot * gaussian_filter(link, sigma) + domain * gaussian_filter(cover + cover.T, sigma)


# ---- Epigenetic checkerboard and its combination with the loop map ----------

def epigenetic_profile(res, n_states: int) -> np.ndarray:
    """Average epigenetic state of every bead over production snapshots, scaled to [-1 (B-like), +1 (A-like)]."""
    half = (n_states - 1) / 2
    return ((res.production(res.epi) - half) / half).mean(axis=1)


STATE_SMOOTH = 2.0      # beads: sliding-window smoothing of the average epigenetic state


def state_profile(res, n_states: int, sigma: float = STATE_SMOOTH) -> np.ndarray:
    """Average Potts state <s>_t of every bead over the production snapshots (A-like = +1, B-like = -1), smoothed along the chain."""
    return gaussian_filter1d(epigenetic_profile(res, n_states), sigma, mode="nearest") if sigma > 0 else epigenetic_profile(res, n_states)


def checkerboard(profile) -> np.ndarray:
    """Outer product of the (unit-peak) average profile: +1 for same-type pairs (A-A, B-B), -1 for A-B."""
    p = np.asarray(profile, float)
    p = p / (np.abs(p).max() or 1.0)
    return np.outer(p, p)


def combine_maps(loops, board, strength: float = 0.5) -> np.ndarray:
    """Loop map times exp(strength * checkerboard): same-type pairs enriched, A-B pairs depleted, always positive."""
    return loops * np.exp(strength * board)


# ---- Hi-C style features of a contact map: PC1 (compartments), insulation (TADs) and P(s) decay ----

@dataclass
class MapFeatures:
    pc1: np.ndarray            # first principal component of the O/E correlation map
    explained: float           # variance share of PC1
    insulation: np.ndarray     # log2 diamond insulation score (NaN near the ends)
    boundaries: np.ndarray     # beads at insulation minima
    s: np.ndarray              # separation (beads)
    ps: np.ndarray             # mean contact at separation s
    slope: float               # log-log slope of P(s)


def observed_over_expected(mat) -> np.ndarray:
    """Divide every diagonal by its mean."""
    n = mat.shape[0]
    out = np.ones_like(mat, dtype=float)
    for k in range(n):
        d = np.diagonal(mat, k)
        m = d.mean()
        idx = np.arange(n - k)
        if m > 0:
            out[idx, idx + k] = out[idx + k, idx] = d / m
    return out


def pc1(mat, ref=None):
    """Leading eigenvector of the Pearson map of O/E; sign chosen to agree with `ref` (e.g. A/B signal)."""
    oe = np.log2(observed_over_expected(mat) + 1e-3)
    c = np.nan_to_num(np.corrcoef(oe))
    vals, vecs = np.linalg.eigh(c)
    v = vecs[:, -1]
    if ref is not None and np.std(ref) > 0 and np.corrcoef(v, ref)[0, 1] < 0:
        v = -v
    return v / (np.abs(v).max() or 1.0), float(vals[-1] / max(vals.sum(), 1e-12))


def compartment_from_hic(hic, min_sep: int = 2) -> np.ndarray:
    """A/B track in [-1, 1] from a Hi-C map (NaN = masked beads): the leading eigenvector of the correlation matrix of the O/E map.

    O/E is clipped to [1/8, 8] and the near-diagonal is left out, so a few extreme pixels cannot dominate. The sign is set so that B
    (negative) is where the short-range contacts are denser; masked beads are interpolated.
    """
    n = len(hic)
    seen = np.isfinite(hic).sum(0) > n / 2
    h = np.clip(fill_masked(hic)[np.ix_(seen, seen)], 0.0, None)
    m = len(h)
    sep = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
    oe = observed_over_expected(h)
    oe[sep < min_sep] = 1.0
    c = np.nan_to_num(np.corrcoef(np.log2(np.clip(oe, 0.125, 8.0))))
    v = (np.linalg.eigh(c)[1][:, -1] if m <= 300 else eigsh(c, k=1, which="LA", v0=np.ones(m))[1][:, 0])
    band = (sep >= min_sep) & (sep <= max(m // 8, 10))
    dens = np.log((h * band).sum(1) / np.maximum(band.sum(1), 1) + 1e-12)       # short-range contact density of every bead
    if np.corrcoef(v, dens)[0, 1] > 0:                                          # denser = B = negative
        v = -v
    lim = 3 * 1.4826 * np.median(np.abs(v - np.median(v))) or np.abs(v).max()    # robust cap before scaling to [-1, 1]
    v = np.clip(v, -lim, lim)
    out = np.interp(np.arange(n), np.flatnonzero(seen), v / (np.abs(v).max() or 1.0))
    return out


def insulation(mat, window: int = 5, prominence: float = 0.5):
    """log2 insulation score (mean contact in a diamond across each bead) and its boundary minima."""
    n = mat.shape[0]
    ins = np.full(n, np.nan)
    for i in range(window, n - window):
        ins[i] = mat[i - window:i, i + 1:i + window + 1].mean()
    eps = 0.02 * np.nanmean(ins)                      # pseudocount: empty diamonds do not give -inf
    score = np.log2((ins + eps) / (np.nanmean(ins) + eps))
    ok = np.nan_to_num(score, nan=0.0)
    peaks, _ = find_peaks(-ok, prominence=prominence * (np.nanstd(score) or 1.0))      # prominence in std units
    return score, peaks[(peaks >= window) & (peaks < n - window)]


def map_features(mat, ref=None, window: int = 5) -> MapFeatures:
    """All features of one contact map."""
    mat = np.asarray(mat, float)
    v, expl = pc1(mat, ref)
    ins, bounds = insulation(mat, window)
    ps = ps_curve(mat)[1:]
    s = np.arange(1, len(ps) + 1)
    ok = (ps > 0) & (s >= 2) & (s <= len(ps) // 2)
    slope = float(np.polyfit(np.log(s[ok]), np.log(ps[ok]), 1)[0]) if ok.sum() > 2 else float("nan")
    return MapFeatures(v, expl, ins, bounds, s, ps, slope)


def compare_features(a: MapFeatures, b: MapFeatures) -> dict:
    """Agreement between two maps: correlation of PC1, insulation and P(s) (log), plus boundary overlap."""
    r = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    ok = ~np.isnan(a.insulation) & ~np.isnan(b.insulation)
    m = (a.ps > 0) & (b.ps > 0)
    near = [np.abs(b.boundaries - x).min() <= 2 for x in a.boundaries] if len(a.boundaries) and len(b.boundaries) else []
    return dict(pc1=abs(r(a.pc1, b.pc1)), insulation=r(a.insulation[ok], b.insulation[ok]),
                ps=r(np.log(a.ps[m]), np.log(b.ps[m])), boundary_overlap=float(np.mean(near)) if near else float("nan"))
