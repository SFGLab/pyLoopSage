"""Resolve the modelled genomic region and bead count."""
from common.constants import CHROM_LENGTHS, DEFAULT_RESOLUTION_BP
from common.geometry import Geometry
from logger import get_logger

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
