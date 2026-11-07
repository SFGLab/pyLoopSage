"""Model inputs from Python variables (no files): loops, peaks and per-bead tracks given as arrays."""
import numpy as np

from common.genome import Geometry
from logger import get_logger
from preproc.loader import InputData
from preproc.readers import _finish, _profiles, loops_from_beads, valid_prob

log = get_logger(__name__)


def geometry(chrom: str, start: int, end: int, n_beads: int = None) -> Geometry:
    """Region in bp and the number of beads (default: about 2 kb per bead)."""
    n = int(n_beads) if n_beads else max(int(round((end - start) / 2000)), 10)
    return Geometry(str(chrom), int(start), int(end), n)


def _beads(geom, pos, units):
    """Positions -> bead indices ('bp': genomic coordinates in the region, 'bead': already bead numbers)."""
    pos = np.asarray(pos, dtype=float)
    if units == "bead":
        return pos.astype(int)
    if units != "bp":
        raise ValueError("units must be 'bp' or 'bead'")
    return np.asarray(geom.bead_of(pos.astype(np.int64)))


def _column(a, n, name, default):
    """Optional 1D array of length n (None -> default)."""
    if a is None:
        return np.full(n, default, dtype=float)
    a = np.asarray(a, dtype=float).ravel()
    if len(a) != n:
        raise ValueError(f"{name} has {len(a)} values, expected {n}")
    return a


def loops_from_arrays(geom, left, right, weight=None, left_reverse=None, right_reverse=None, units="bp",
                      smooth=False, contrastive=True, sigma=1.0, loss_sigma=0.0):
    """Loops from arrays: anchor positions `left`, `right` (bp or bead numbers), optional `weight` (e.g. PET counts) and
    optional probabilities that the CTCF at each anchor points to the right (`left_reverse`, `right_reverse`, in [0, 1]).
    Loops outside the region are dropped. Returns LoopData."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left and right must be 1D arrays of the same length")
    n = len(left)
    w = _column(weight, n, "weight", 1.0)
    p1 = p2 = None
    if left_reverse is not None or right_reverse is not None:
        p1, p2 = _column(left_reverse, n, "left_reverse", -1.0), _column(right_reverse, n, "right_reverse", -1.0)
        if not (valid_prob(p1) and valid_prob(p2)):
            raise ValueError("left_reverse and right_reverse must be probabilities in [0, 1] (or -1 for unknown)")
    if units == "bp":                                # keep only loops whose anchors lie inside the region
        inside = (left >= geom.start) & (left < geom.end) & (right >= geom.start) & (right < geom.end)
        left, right, w = left[inside], right[inside], w[inside]
        p1, p2 = (None, None) if p1 is None else (p1[inside], p2[inside])
    return loops_from_beads(geom, _beads(geom, left, units), _beads(geom, right, units), w, p1, p2, smooth, contrastive, sigma, loss_sigma)


def sites_from_arrays(geom, position, weight=None, reverse=None, units="bp", smooth=False, sigma=2.0):
    """Binding sites (peaks) from arrays: `position`, optional `weight` and probability `reverse` that the site points to the right."""
    pos = np.asarray(position)
    if units == "bp":
        pos = pos[(pos >= geom.start) & (pos < geom.end)]
    n = len(pos)
    if n == 0:
        raise ValueError(f"no sites in {geom.label}")
    w = _column(weight, n, "weight", 1.0)
    prob = _column(reverse, n, "reverse", -1.0)
    if not valid_prob(prob):
        raise ValueError("reverse must be probabilities in [0, 1] (or -1 for unknown)")
    L, R = _profiles(_beads(geom, pos, units), w, prob, None, geom, smooth, sigma, False)
    return _finish(L, R, w, prob, geom, "sites")


def inputs_from_arrays(geom, loops=None, sites=None, tracks=None, track_names=None, lef_track=None, compartments=None, **options) -> InputData:
    """All model inputs from Python variables.

    loops:        a dict for `loops_from_arrays` (left, right, weight, ...) or a tuple (left, right[, weight]); or a LoopData
    sites:        same for `sites_from_arrays` (position, weight, reverse); used when there are no loops
    tracks:       extra binding tracks, array (n_tracks, n_beads) or a list of per-bead arrays
    lef_track:    preference of each bead for loading a LEF (per-bead array, >= 0)
    compartments: A/B signal per bead; values outside [-1, 1] are rescaled to it (B negative)
    options:      smooth, contrastive, ... passed on to the loop or site builder
    """
    n = geom.n_beads
    if loops is not None:
        if isinstance(loops, dict):
            data = loops_from_arrays(geom, **{**options, **loops})
        elif isinstance(loops, (tuple, list)):
            data = loops_from_arrays(geom, *loops, **options)
        else:
            data = loops
    elif sites is not None:
        data = sites_from_arrays(geom, **{**{k: v for k, v in options.items() if k in ("units", "smooth", "sigma")}, **sites}) if isinstance(sites, dict) else sites_from_arrays(geom, *sites)
    else:
        raise ValueError("give loops or sites")
    tr = np.zeros((0, n)) if tracks is None else np.atleast_2d(np.asarray(tracks, dtype=float))
    if tr.shape[1] != n:
        raise ValueError(f"tracks must have {n} values per track (one per bead), got {tr.shape[1]}")
    names = list(track_names) if track_names is not None else [f"track{i + 1}" for i in range(len(tr))]
    lef = None if lef_track is None else np.clip(_column(lef_track, n, "lef_track", 0.0), 0, None)
    comp = None
    if compartments is not None:
        comp = _column(compartments, n, "compartments", 0.0)
        if np.abs(comp).max() > 1:
            lo, hi = comp.min(), comp.max()
            comp = 2 * (comp - lo) / (hi - lo) - 1
    return InputData(data, tr, names, lef, comp, "array" if comp is not None else "")
