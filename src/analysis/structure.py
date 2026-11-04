"""Per-frame shape metrics of a 3D chromatin structure (all lengths in nm).

- Pair distances, packing, scaling and fractal dimension of 3D structures (nm).
- All 3D structure diagnostics in one call, plus the summary table rows.
- 3D geometry of the MD ensemble: polymer scaling and loop-anchor proximity.
- Does the force field work? Bond, angle, excluded-volume and loop-spring diagnostics from MD frames (nm).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist, squareform

from analysis.ensemble import loop_pairs_of_frame
from molecular_dynamics.forces import BOND_K, BOND_LENGTH
from molecular_dynamics.structures import align_frames


# ---- Per-frame shape metrics of a 3D chromatin structure (all lengths in nm) ----

def gyration_eigs(c):
    """Eigenvalues (descending) of the gyration tensor, in nm^2."""
    d = c - c.mean(axis=0)
    return np.sort(np.linalg.eigvalsh(d.T @ d / len(c)))[::-1]


def shape_metrics(c) -> dict:
    """Radius of gyration, asphericity, acylindricity, relative shape anisotropy, end-to-end, convex hull."""
    l1, l2, l3 = gyration_eigs(c)
    rg2 = l1 + l2 + l3
    b, cyl = l1 - 0.5 * (l2 + l3), l2 - l3
    try:
        hull = ConvexHull(c)
        vol, area = hull.volume, hull.area
    except QhullError:
        vol = area = np.nan
    return dict(rg=np.sqrt(rg2), asphericity=b, acylindricity=cyl,
                anisotropy=(b ** 2 + 0.75 * cyl ** 2) / rg2 ** 2 if rg2 > 0 else 0.0,   # kappa^2: 0 sphere, 1 rod
                end_to_end=np.linalg.norm(c[-1] - c[0]), hull_volume=vol, hull_area=area,
                sphericity=np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area if area else np.nan)


def shape_series(coords) -> dict:
    """Metrics for every frame: {name: array(frames)}."""
    rows = [shape_metrics(c) for c in coords]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ---- Pair distances, packing, scaling and fractal dimension of 3D structures (nm) ----

def bond_stats(coords) -> dict:
    """Chain integrity: mean bond length and its coefficient of variation per frame."""
    b = np.linalg.norm(np.diff(coords, axis=1), axis=2)
    return dict(bond_mean=b.mean(axis=1), bond_cv=b.std(axis=1) / b.mean(axis=1))


def pair_metrics(c, radius: float) -> dict:
    """Mean pair distance, nearest non-bonded neighbour, closest non-bonded pair and contacts per bead."""
    d = squareform(pdist(c))
    n = len(c)
    far = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 2          # exclude chain neighbours
    nb = np.where(far, d, np.inf)
    return dict(mean_pair=d[np.triu_indices(n, 1)].mean(), nearest=nb.min(axis=1).mean(), closest=nb.min(),
                contacts=((nb < radius).sum() / n))


def pair_distribution(coords, bins) -> np.ndarray:
    """Normalised histogram of all non-bonded pair distances pooled over frames."""
    h = np.zeros(len(bins) - 1)
    for c in coords:
        d = pdist(c)
        i, j = np.triu_indices(len(c), 1)
        h += np.histogram(d[(j - i) >= 2], bins=bins)[0]
    return h / h.sum() / np.diff(bins)


def radial_density(coords, edges) -> np.ndarray:
    """Bead number density (1/nm^3) in spherical shells around the centre of mass, averaged over frames."""
    counts = np.zeros(len(edges) - 1)
    for c in coords:
        counts += np.histogram(np.linalg.norm(c - c.mean(axis=0), axis=1), bins=edges)[0]
    return counts / len(coords) / (4 / 3 * np.pi * np.diff(edges ** 3))


def mass_radius_dimension(c, lo: float = 0.15, hi: float = 0.9) -> float:
    """Fractal dimension d_f from N(r) ~ r^d_f (beads within r of the centre of mass)."""
    r = np.sort(np.linalg.norm(c - c.mean(axis=0), axis=1))
    sel = (r > lo * r[-1]) & (r < hi * r[-1])
    n = np.arange(1, len(r) + 1)
    return float(np.polyfit(np.log(r[sel]), np.log(n[sel]), 1)[0]) if sel.sum() > 3 else np.nan


def scaling_exponent(c) -> tuple:
    """(s, R(s), nu): mean 3D distance between beads s apart and its exponent (R ~ s^nu); d_f = 1/nu."""
    n = len(c)
    s = np.arange(1, n // 2)
    r = np.array([np.linalg.norm(c[k:] - c[:-k], axis=1).mean() for k in s])
    fit = (s >= 5) & (s <= max(n // 4, 6))
    nu = float(np.polyfit(np.log(s[fit]), np.log(r[fit]), 1)[0]) if fit.sum() > 2 else np.nan
    return s, r, nu


def rmsd(a, b) -> float:
    """Root-mean-square deviation of two aligned structures."""
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


# ---- All 3D structure diagnostics in one call, plus the summary table rows ----

_MAX_PAIR_FRAMES = 60      # pair-based metrics are O(N^2) per frame: use an even subset


@dataclass
class StructureMetrics:
    shape: dict                     # per-frame shape metrics (see structure_shape)
    bonds: dict                     # per-frame bond statistics
    pairs: dict                     # per-frame pair metrics (on the frame subset)
    pair_frames: np.ndarray         # frame indices of the subset
    dist_bins: np.ndarray
    dist_pdf: np.ndarray            # pooled pair-distance distribution (1/nm)
    r_edges: np.ndarray
    density: np.ndarray             # radial density profile (1/nm^3)
    s: np.ndarray                   # genomic separation (beads)
    r_of_s: np.ndarray              # mean 3D distance at each separation (nm)
    nu: np.ndarray                  # per-frame scaling exponent (subset)
    df_mass: np.ndarray             # per-frame mass-radius dimension (subset)
    rmsd_step: np.ndarray           # RMSD between consecutive aligned frames (nm)
    compart: dict = field(default_factory=dict)   # radial position / Rg of A and B beads per frame


def _compartments(coords, comp) -> dict:
    if comp is None:
        return {}
    a, b = np.asarray(comp) >= 0, np.asarray(comp) < 0
    if a.sum() < 3 or b.sum() < 3:
        return {}
    out = {k: [] for k in ("r_a", "r_b", "rg_a", "rg_b")}
    for c in coords:
        r = np.linalg.norm(c - c.mean(axis=0), axis=1)
        out["r_a"].append(r[a].mean()); out["r_b"].append(r[b].mean())
        out["rg_a"].append(np.sqrt(((c[a] - c[a].mean(axis=0)) ** 2).sum(axis=1).mean()))
        out["rg_b"].append(np.sqrt(((c[b] - c[b].mean(axis=0)) ** 2).sum(axis=1).mean()))
    return {k: np.array(v) for k, v in out.items()}


def compute_metrics(coords, contact_radius: float, comp=None) -> StructureMetrics:
    """Shape, packing, scaling, fractal and stability metrics of an MD ensemble (coords in nm)."""
    coords = np.asarray(coords, dtype=float)
    sub = np.unique(np.linspace(0, len(coords) - 1, min(_MAX_PAIR_FRAMES, len(coords))).astype(int))
    rows = [pair_metrics(coords[k], contact_radius) for k in sub]
    pairs = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    top = float(np.percentile([np.linalg.norm(c - c.mean(axis=0), axis=1).max() for c in coords[sub]], 95)) * 2
    bins, edges = np.linspace(0, top, 60), np.linspace(0, top / 2, 25)
    scal = [scaling_exponent(coords[k]) for k in sub]
    aligned = align_frames(coords)
    return StructureMetrics(
        shape=shape_series(coords), bonds=bond_stats(coords), pairs=pairs, pair_frames=sub, dist_bins=bins,
        dist_pdf=pair_distribution(coords[sub], bins), r_edges=edges, density=radial_density(coords[sub], edges),
        s=scal[0][0], r_of_s=np.mean([x[1] for x in scal], axis=0), nu=np.array([x[2] for x in scal]),
        df_mass=np.array([mass_radius_dimension(coords[k]) for k in sub]),
        rmsd_step=np.array([rmsd(aligned[k], aligned[k - 1]) for k in range(1, len(coords))]),
        compart=_compartments(coords, comp))


def summary_rows(m: StructureMetrics) -> list:
    """[metric, mean, std, unit, note] rows for the diagnostics table."""
    row = lambda name, v, unit, note="": [name, float(np.nanmean(v)), float(np.nanstd(v)), unit, note]
    s, p, b = m.shape, m.pairs, m.bonds
    nu = np.nanmean(m.nu)
    return [
        row("Radius of gyration", s["rg"], "nm", "overall size"),
        row("End-to-end distance", s["end_to_end"], "nm"),
        row("Asphericity", s["asphericity"], "nm^2", "0 = spherical"),
        row("Acylindricity", s["acylindricity"], "nm^2", "0 = cylindrical symmetry"),
        row("Relative shape anisotropy", s["anisotropy"], "(0-1)", "0 sphere, 1 rod"),
        row("Convex hull volume", s["hull_volume"], "nm^3", "space occupied"),
        row("Convex hull surface", s["hull_area"], "nm^2"),
        row("Hull sphericity", s["sphericity"], "(0-1)", "1 = perfect sphere"),
        row("Bond length", b["bond_mean"], "nm", "chain integrity"),
        row("Bond length variation", b["bond_cv"], "(CV)"),
        row("Mean pair distance", p["mean_pair"], "nm"),
        row("Nearest non-bonded neighbour", p["nearest"], "nm", "packing"),
        row("Closest non-bonded pair", p["closest"], "nm", "overlap check"),
        row("Contacts per bead", p["contacts"], "count", "non-bonded, within contact radius"),
        row("Scaling exponent nu (R ~ s^nu)", m.nu, "(dimensionless)", "1/3 globule, 1/2 ideal, 0.59 swollen"),
        row("Fractal dimension, 1/nu", 1 / m.nu, "(dimensionless)", "from scaling"),
        row("Fractal dimension, mass-radius", m.df_mass, "(dimensionless)", "N(r) ~ r^d_f; 3 = space filling"),
        row("Frame-to-frame RMSD", m.rmsd_step, "nm", "structural change between frames") if len(m.rmsd_step) else
        ["Frame-to-frame RMSD", float("nan"), float("nan"), "nm", ""],
    ] + ([row("Mean radial position, A beads", m.compart["r_a"], "nm", "distance from centre of mass"),
          row("Mean radial position, B beads", m.compart["r_b"], "nm")] if m.compart else [])


# ---- 3D geometry of the MD ensemble: polymer scaling and loop-anchor proximity ----

def distance_scaling(coords, max_frames: int = 20):
    """Mean spatial distance R(s) (nm) between beads s apart, and the exponent nu in R ~ s^nu."""
    c = coords[np.linspace(0, len(coords) - 1, min(max_frames, len(coords))).astype(int)]
    n = c.shape[1]
    s = np.arange(1, n // 2)
    r = np.array([np.linalg.norm(c[:, k:] - c[:, :-k], axis=-1).mean() for k in s])
    fit = (s >= 5) & (s <= max(n // 4, 6))
    nu = float(np.polyfit(np.log(s[fit]), np.log(r[fit]), 1)[0]) if fit.sum() > 2 else np.nan
    return s, r, nu


def anchor_distances(md, mc, seed: int = 0):
    """Spatial distance (nm) of loop anchors, and of random bead pairs with the same genomic separation."""
    rng, loop, ctrl = np.random.default_rng(seed), [], []
    n = md.coords.shape[1]
    for k, i in enumerate(md.frame_idx):
        m, nn = np.minimum(mc.Ms[:, i], mc.Ns[:, i]), np.maximum(mc.Ms[:, i], mc.Ns[:, i])
        ok = nn > m
        m, nn = m[ok], nn[ok]
        c = md.coords[k]
        loop.append(np.linalg.norm(c[m] - c[nn], axis=1))
        start = (rng.random(len(m)) * (n - (nn - m))).astype(int)
        ctrl.append(np.linalg.norm(c[start] - c[start + (nn - m)], axis=1))
    return np.concatenate(loop), np.concatenate(ctrl)


# ---- Does the force field work? Bond, angle, excluded-volume and loop-spring diagnostics from MD frames (nm) ----

KB = 0.0083144626      # kJ/mol/K
MIN_SEP = 3            # non-bonded: at least this many beads apart along the chain


@dataclass
class ForceDiag:
    bond: np.ndarray            # (F, N-1) bond lengths
    angle: np.ndarray           # (F, N-2) bend angles in degrees (180 = straight)
    kT: float
    k_angle: float
    k_loop: float
    loop_rest: float
    g_edges: np.ndarray         # radial distribution of non-bonded pairs
    g: np.ndarray
    nearest: np.ndarray         # nearest non-bonded neighbour of every bead (pooled)
    overlap: np.ndarray         # per frame: fraction of non-bonded pairs closer than half a bond
    closest: np.ndarray         # per frame: closest non-bonded pair
    anchor_mean: np.ndarray     # per frame loop-anchor distance: mean, 25%, 75%
    anchor_q: np.ndarray
    anchor_all: np.ndarray      # pooled anchor distances
    gap: np.ndarray             # inter-loop: genomic gap (beads) and 3D distance (nm) of neighbouring loops
    dist: np.ndarray


def _bend(c):
    a, b = c[:-2] - c[1:-1], c[2:] - c[1:-1]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def force_diagnostics(md, mc, cfg, max_frames: int = 60) -> ForceDiag:
    """Pool bonds/angles over all frames; pair-based quantities use an even subset of frames."""
    coords = np.asarray(md.coords, float)
    sub = np.unique(np.linspace(0, len(coords) - 1, min(max_frames, len(coords))).astype(int))
    n = coords.shape[1]
    bond = np.linalg.norm(np.diff(coords, axis=1), axis=2)
    angle = np.array([_bend(c) for c in coords])
    edges = np.linspace(0, 6 * BOND_LENGTH + 0.4, 50)
    cnt, nearest, overlap, closest = np.zeros(len(edges) - 1), [], [], []
    iu = np.triu_indices(n, MIN_SEP)
    mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= MIN_SEP
    for k in range(len(coords)):
        d = squareform(pdist(coords[k]))
        far = d[iu]
        overlap.append((far < 0.5 * BOND_LENGTH).mean())
        closest.append(far.min())
        if k in sub:
            cnt += np.histogram(far, edges)[0]
            nearest.append(np.where(mask, d, np.inf).min(1))
    mid = (edges[1:] + edges[:-1]) / 2
    g = cnt / (len(sub) * len(far)) / (4 * np.pi * mid ** 2 * np.diff(edges))          # ~ pair density; flat = ideal gas
    am, aq, allv, gap, dist = [], [], [], [], []
    for k, i in enumerate(md.frame_idx):
        loops = loop_pairs_of_frame(mc.Ms[:, i], mc.Ns[:, i])
        if not len(loops):
            am.append(np.nan); aq.append([np.nan, np.nan])
            continue
        c = coords[k]
        a = np.linalg.norm(c[loops[:, 0]] - c[loops[:, 1]], axis=1)
        am.append(a.mean()); aq.append(np.percentile(a, [25, 75])); allv.append(a)
        if k in sub and len(loops) > 1:
            mid_b = (loops[:, 0] + loops[:, 1]) / 2
            order = np.argsort(mid_b)
            for x, y in zip(order[:-1], order[1:]):                    # neighbouring loops along the chain
                if loops[x, 1] < loops[y, 0]:
                    gap.append(loops[y, 0] - loops[x, 1]); dist.append(np.linalg.norm(c[loops[x, 1]] - c[loops[y, 0]]))
    kT = KB * float(md.meta.get("temperature_K", cfg.sim_temp))
    return ForceDiag(bond, angle, kT, cfg.angle_ff_strength, cfg.le_ff_strength, cfg.le_ff_length, mid, g, np.concatenate(nearest),
                     np.array(overlap), np.array(closest), np.array(am), np.array(aq), np.concatenate(allv) if allv else np.zeros(0),
                     np.array(gap), np.array(dist))


def force_rows(f: ForceDiag) -> list:
    """[quantity, measured, expected, unit] rows for a table."""
    sd_bond = np.sqrt(f.kT / BOND_K)
    return [
        ["Bond length, mean", float(f.bond.mean()), BOND_LENGTH, "nm"],
        ["Bond length, std", float(f.bond.std()), float(sd_bond), "nm (thermal: sqrt(kT/k))"],
        ["Bend angle, mean", float(f.angle.mean()), 180.0, "deg (straight = 180)"],
        ["Bend angle, spread (std)", float(f.angle.std()), float(np.degrees(np.sqrt(f.kT / f.k_angle))), "deg (rough thermal estimate)"],
        ["Non-bonded pairs closer than half a bond", float(f.overlap.mean()), 0.0, "fraction (excluded volume)"],
        ["Closest non-bonded pair", float(f.closest.min()), BOND_LENGTH, "nm"],
        ["Loop-anchor distance, median", float(np.median(f.anchor_all)) if len(f.anchor_all) else float("nan"), f.loop_rest, "nm (spring rest length)"],
    ]
