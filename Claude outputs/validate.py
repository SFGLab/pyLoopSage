"""Sanity checks on the final configuration."""
from pathlib import Path


def validate(cfg) -> None:
    """Raise ValueError with a clear message for the first problem found."""
    if not cfg.interaction_file:
        raise ValueError("an input file is required (-f FILE or interaction_file in config.ini)")
    if not Path(cfg.interaction_file).is_file():
        raise ValueError(f"input file not found: {cfg.interaction_file}")
    if not cfg.chrom:
        raise ValueError("a chromosome is required (-r CHR:START-END or chrom in config.ini)")
    if cfg.mc_step < 1 or cfg.n_steps < cfg.mc_step:
        raise ValueError("need 1 <= sample-every <= mc-steps")
    if cfg.burnin >= cfg.n_steps:
        raise ValueError(f"burn-in ({cfg.burnin}) must be smaller than mc-steps ({cfg.n_steps})")
    if cfg.comp_bw_file and cfg.comp_bed_file:
        raise ValueError("give only one compartment file (BigWig or BED)")
    if cfg.bw_strengths and len(cfg.bw_strengths) != len(cfg.bw_files):
        raise ValueError("bw_strengths must have one value per bw_files entry")
