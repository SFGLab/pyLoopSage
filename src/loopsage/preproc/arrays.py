"""Model inputs from Python variables (no files): loops, peaks and per-bead tracks given as arrays."""
from pathlib import Path

import numpy as np

from loopsage.common.genome import Geometry
from loopsage.logger import get_logger
from loopsage.preproc.loader import InputData
from loopsage.preproc.readers import _finish, _profiles, binding_track, loops_from_beads, valid_prob

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


def check_compartments(values, n) -> np.ndarray:
    """Compartment signal as one finite value in [-1, 1] per bead (A positive, B negative); anything else is an error."""
    comp = _column(values, n, "compartments", 0.0)
    if not np.isfinite(comp).all() or np.abs(comp).max() > 1:
        raise ValueError(f"compartments must be finite values between -1 and 1 (A positive, B negative); got range "
                         f"[{np.nanmin(comp):.3g}, {np.nanmax(comp):.3g}]. Rescale them first, e.g. 2 * (x - x.min()) / (x.max() - x.min()) - 1")
    return comp


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


def inputs_from_arrays(geom, loops=None, sites=None, tracks=None, track_names=None, lef_track=None, compartments=None, track_sigma=0.0, **options) -> InputData:
    """All model inputs from Python variables.

    loops:        a dict for `loops_from_arrays` (left, right, weight, ...) or a tuple (left, right[, weight]); or a LoopData
    sites:        same for `sites_from_arrays` (position, weight, reverse); used when there are no loops
    tracks:       extra binding tracks, array (n_tracks, n_beads) or a list of per-bead arrays
    lef_track:    preference of each bead for loading a LEF (per-bead array, >= 0)
    compartments: A/B signal per bead; values outside [-1, 1] are rescaled to it (B negative)
    options:      smooth, contrastive, ... passed on to the loop or site builder; `track_sigma` smooths the tracks (beads, default 0)
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
    tr = np.array([binding_track(v, track_sigma, nm) for v, nm in zip(tr, names)]).reshape(len(tr), n)
    lef = None if lef_track is None else np.clip(_column(lef_track, n, "lef_track", 0.0), 0, None)
    comp = None
    if compartments is not None:
        comp = check_compartments(compartments, n)
    return InputData(data, tr, names, lef, comp, "array" if comp is not None else "")


# ---- Variables -> temporary files, so every stage works the same as with files ----

def _bp(geom, pos, units):
    """Positions as integer bp (bead numbers become bead centres)."""
    pos = np.asarray(pos, dtype=float)
    if units == "bead":
        return (geom.start + pos * geom.bp_per_bead + geom.bp_per_bead // 2).astype(np.int64)
    if units != "bp":
        raise ValueError("units must be 'bp' or 'bead'")
    return pos.astype(np.int64)


def write_loops_bedpe(path, geom, left, right, weight=None, left_reverse=None, right_reverse=None, units="bp"):
    """Write loops given as arrays to a BEDPE file (anchors are one-bp intervals; orientation columns when given)."""
    left, right = _bp(geom, left, units), _bp(geom, right, units)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left and right must be 1D arrays of the same length")
    n = len(left)
    w = _column(weight, n, "weight", 1.0)
    cols = [f"{geom.chrom}\t{a}\t{a + 1}\t{geom.chrom}\t{b}\t{b + 1}\t{x:g}" for a, b, x in zip(left, right, w)]
    if left_reverse is not None or right_reverse is not None:
        p1, p2 = _column(left_reverse, n, "left_reverse", -1.0), _column(right_reverse, n, "right_reverse", -1.0)
        cols = [f"{c}\t{a:g}\t{b:g}" for c, a, b in zip(cols, p1, p2)]
    Path(path).write_text("\n".join(cols) + "\n")


def write_sites_bed(path, geom, position, weight=None, reverse=None, units="bp"):
    """Write binding sites given as arrays to a BED file (column 5 = weight, column 7 = reverse-orientation probability)."""
    pos = _bp(geom, position, units)
    n = len(pos)
    w, prob = _column(weight, n, "weight", 1.0), _column(reverse, n, "reverse", -1.0)
    Path(path).write_text("".join(f"{geom.chrom}\t{a}\t{a + 1}\tsite{i + 1}\t{x:g}\t.\t{q:g}\n" for i, (a, x, q) in enumerate(zip(pos, w, prob))))


def patch_inputs(inputs, geom, tracks=None, track_names=None, lef_track=None, compartments=None, track_sigma=0.0):
    """Put per-bead arrays (extra tracks, LEF loading preference, compartments) into an InputData that was read from files."""
    n = geom.n_beads
    if tracks is not None:
        tr = np.atleast_2d(np.asarray(tracks, dtype=float))
        if tr.shape[1] != n:
            raise ValueError(f"tracks must have {n} values per track (one per bead), got {tr.shape[1]}")
        names = list(track_names) if track_names is not None else [f"track{i + 1}" for i in range(len(tr))]
        tr = np.array([binding_track(v, track_sigma, nm) for v, nm in zip(tr, names)])
        inputs.BWs = np.vstack([inputs.BWs.reshape(-1, n), tr])
        inputs.bw_names = list(inputs.bw_names) + names
    if lef_track is not None:
        inputs.lef_track = np.clip(_column(lef_track, n, "lef_track", 0.0), 0, None)
    if compartments is not None:
        comp = check_compartments(compartments, n)
        inputs.comp, inputs.comp_source = comp, "array"
    return inputs
