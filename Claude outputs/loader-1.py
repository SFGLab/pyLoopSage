"""Load every model input for the configured region."""
from pathlib import Path

import numpy as np

from logger import get_logger
from preproc.bedpe import vectors_from_bedpe
from preproc.compartments import load_compartments_bed
from preproc.peaks import vectors_from_bed, vectors_from_narrowpeak
from preproc.tracks import bigwig_to_beads
from preproc.types import InputData

log = get_logger(__name__)

_READERS = {".bedpe": vectors_from_bedpe, ".bed": vectors_from_bed, ".narrowpeak": vectors_from_narrowpeak}


def _compartments(cfg, geom):
    if cfg.comp_bw_file:
        return bigwig_to_beads(cfg.comp_bw_file, geom, scale_pm1=True)
    if cfg.comp_bed_file:
        return load_compartments_bed(cfg.comp_bed_file, geom, spline_s=geom.n_beads / 100)
    return None


def load_inputs(cfg, geom) -> InputData:
    """Dispatch on the interaction-file extension and attach optional tracks."""
    ext = Path(cfg.interaction_file).suffix.lower()
    if ext not in _READERS:
        raise ValueError(f"unsupported input format '{ext}' (use .bedpe, .bed or .narrowPeak)")
    loops = _READERS[ext](cfg.interaction_file, geom, smooth=cfg.smoothing_input,
                          contrastive=cfg.contrastive_binding, sigma=geom.n_beads / 100)
    bws = [bigwig_to_beads(f, geom) for f in cfg.bw_files]
    lef = None
    if cfg.lef_track_file:
        lef = np.clip(bigwig_to_beads(cfg.lef_track_file, geom, roll=True), 0, None)
    log.info("Loaded %d extra track(s), LEF track: %s, compartments: %s",
             len(bws), lef is not None, bool(cfg.comp_bw_file or cfg.comp_bed_file))
    return InputData(loops, np.array(bws).reshape(len(bws), geom.n_beads),
                     [Path(f).name for f in cfg.bw_files], lef, _compartments(cfg, geom))
