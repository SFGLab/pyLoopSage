"""Containers for preprocessed model inputs (with disk cache).

- Resolve the modelled genomic region and bead count.
- Compartment BED (A/B sub-compartments) -> signal in [-1, 1] per bead.
- Load every model input for the configured region.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from common.genome import CHROM_LENGTHS, DEFAULT_RESOLUTION_BP, Geometry
from logger import get_logger
from preproc.readers import LoopData, bigwig_to_beads, vectors_from_bed, vectors_from_bedpe, vectors_from_narrowpeak


# ---- Containers for preprocessed model inputs (with disk cache) ------------

@dataclass
class InputData:
    loops: LoopData
    BWs: np.ndarray                  # (n_tracks, N) extra binding tracks
    bw_names: list = field(default_factory=list)
    lef_track: np.ndarray = None     # LEF loading preference per bead
    comp: np.ndarray = None          # compartment signal in [-1, 1] per bead

    def save(self, path) -> None:
        lo = self.loops
        arrays = dict(L=lo.L, R=lo.R, J=lo.J, J_loss=lo.J_loss, lengths=lo.loop_lengths, BWs=self.BWs)
        for name in ("lef_track", "comp"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        meta = json.dumps(dict(n_sites=int(lo.n_sites), stats=lo.stats, fmt=lo.fmt, bw_names=self.bw_names))
        np.savez_compressed(path, meta=np.array(meta), **arrays)

    @classmethod
    def load(cls, path) -> "InputData":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        loops = LoopData(z["L"], z["R"], z["J"], z["J_loss"], meta["n_sites"], z["lengths"], meta["stats"], meta["fmt"])
        return cls(loops, z["BWs"], meta["bw_names"],
                   z["lef_track"] if "lef_track" in z else None, z["comp"] if "comp" in z else None)


# ---- Resolve the modelled genomic region and bead count --------------------

log = get_logger(__name__)


def build_geometry(cfg) -> Geometry:
    """Region (bp) + beads from the config; falls back to the whole chromosome."""
    sizes = CHROM_LENGTHS[cfg.genome]
    if cfg.chrom not in sizes:
        raise ValueError(f"unknown chromosome '{cfg.chrom}' for {cfg.genome}")
    size = sizes[cfg.chrom]
    start, end = cfg.region_start, cfg.region_end
    if start is None or end is None or start < 0 or end <= start:
        log.warning("No valid region given: using the whole %s (%d bp).", cfg.chrom, size)
        start, end = 0, size
    if end > size:
        log.warning("Region end %d exceeds %s length %d: clipped.", end, cfg.chrom, size)
        end = size
    n_beads = cfg.n_beads or max(int(round((end - start) / DEFAULT_RESOLUTION_BP)), 10)
    geom = Geometry(cfg.chrom, int(start), int(end), int(n_beads))
    log.info("Region %s -> %d beads of %d bp", geom.label, geom.n_beads, geom.bp_per_bead)
    return geom


# ---- Compartment BED (A/B sub-compartments) -> signal in [-1, 1] per bead ----

def _label_value(label) -> float:
    """'A.1.2' -> +1.2, 'B.1' -> -1.0: sign from A/B, magnitude grows with depth."""
    if not isinstance(label, str) or label[:1] not in "AB":
        return 0.0
    return (1.0 if label[0] == "A" else -1.0) * (1.0 + 0.2 * (len(label.split(".")) - 1))


def _values(df, use_score: bool) -> np.ndarray:
    """Per-row value: 2*score-1 if the score column is numeric, else from the label."""
    by_label = df[3].map(_label_value).to_numpy(dtype=float)
    if not use_score or df.shape[1] < 5:
        return by_label
    score = pd.to_numeric(df[4], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isnan(score), by_label, 2.0 * score - 1.0)


def load_compartments_bed(path, geom, use_score=True, spline_s=None) -> np.ndarray:
    """Average row values per bead, optional spline smoothing, genome-wide [-1, 1] scaling."""
    full = pd.read_csv(path, sep="\t", header=None)
    vals = _values(full, use_score)
    gmin, gmax = vals.min(), vals.max()              # global reference keeps scale comparable
    sel = (full[0] == geom.chrom) & (full[1] < geom.end) & (full[2] > geom.start)
    if not sel.any():
        raise ValueError(f"no compartments found in {geom.label}")
    df, v = full[sel].reset_index(drop=True), vals[sel.to_numpy()]

    n = geom.n_beads
    signal, count = np.zeros(n), np.zeros(n)
    b0 = geom.bead_of(np.maximum(df[1].to_numpy(), geom.start))
    b1 = geom.bead_of(np.minimum(df[2].to_numpy(), geom.end))
    for lo, hi, val in zip(b0, b1, v):
        signal[lo:hi + 1] += val
        count[lo:hi + 1] += 1
    covered = count > 0
    signal[covered] /= count[covered]

    if spline_s is not None and covered.sum() > 3:   # fill gaps, then smooth
        x = np.arange(n)
        signal[~covered] = np.interp(x[~covered], x[covered], signal[covered])
        signal = UnivariateSpline(x, signal, s=spline_s)(x)
    return 2.0 * (signal - gmin) / (gmax - gmin) - 1.0 if gmax > gmin else np.zeros(n)


# ---- Load every model input for the configured region ----------------------


_READERS = {".bedpe": vectors_from_bedpe, ".bed": vectors_from_bed, ".narrowpeak": vectors_from_narrowpeak}


def _compartments(cfg, geom):
    if cfg.comp_bw_file:
        return bigwig_to_beads(cfg.comp_bw_file, geom, scale_pm1=True)
    if cfg.comp_bed_file:
        return load_compartments_bed(cfg.comp_bed_file, geom, spline_s=geom.n_beads / 100)
    return None


def load_inputs(cfg, geom, path=None) -> InputData:
    """Dispatch on the interaction-file extension and attach optional tracks (path overrides cfg.interaction_file)."""
    path = path or cfg.interaction_file
    ext = Path(path).suffix.lower()
    if ext not in _READERS:
        raise ValueError(f"unsupported input format '{ext}' (use .bedpe, .bed or .narrowPeak)")
    loops = _READERS[ext](path, geom, smooth=cfg.smoothing_input,
                          contrastive=cfg.contrastive_binding, sigma=geom.n_beads / 100)
    bws = [bigwig_to_beads(f, geom) for f in cfg.bw_files]
    lef = None
    if cfg.lef_track_file:
        lef = np.clip(bigwig_to_beads(cfg.lef_track_file, geom, roll=True), 0, None)
    log.info("Loaded %d extra track(s), LEF track: %s, compartments: %s",
             len(bws), lef is not None, bool(cfg.comp_bw_file or cfg.comp_bed_file))
    return InputData(loops, np.array(bws).reshape(len(bws), geom.n_beads),
                     [Path(f).name for f in cfg.bw_files], lef, _compartments(cfg, geom))
