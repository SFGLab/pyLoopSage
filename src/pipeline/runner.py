"""State shared by all stages of one run.

- Chain the stages; each one computes (or loads a cached result) and describes it.
"""
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.genome import fmt_duration, Geometry
from common.runtime import apply_threads, create_run_dirs, OutputDirs
from common.utils import Timer
from knots import stage as stage_knots
from logger import get_logger, print_banner, set_quiet, setup_logging
from pipeline import stage_analysis, stage_mc, stage_md, stage_optimize, stage_preproc, stage_report, stage_structure
from preproc.loader import build_geometry
from reporting.report import REGISTRY
from settings.loader import load_config, write_ini


# ---- State shared by all stages of one run ---------------------------------

@dataclass
class RunContext:
    cfg: Any
    dirs: OutputDirs
    geom: Geometry
    inputs: Any = None       # preproc.types.InputData
    motif: Any = None        # motif_finder.MotifResult (when the CTCF motif search ran)
    input_file: Any = None   # loops file actually used (annotated BEDPE if the motif search ran)
    mc: Any = None           # monte_carlo.result.MCResult
    mc_map: Any = None       # contact map implied by the MC loops (times the epigenetic checkerboard when enabled)
    mc_loop_map: Any = None  # loop-only map, set when the epigenetic model is on
    md: Any = None           # molecular_dynamics.result.MDResult
    structure: Any = None    # pipeline.stage_structure.StructureResult
    validation: Any = None   # validation.compare.ValidationResult
    features: Any = None     # {"mc": MapFeatures, "md": MapFeatures}
    links: Any = None        # LinkResult (optional knots stage)
    report_path: Any = None
    _pool: Any = None
    optimization: Any = None   # optimizer.nelder_mead.OptResult (with --optimize)
    mc_epi_profile: Any = None   # average epigenetic state per bead (MC compartment track), None when the epigenetic model is off
    val_source: str = ""     # which contact map was validated
    _hic: Any = None

    @property
    def loops_file(self) -> str:
        """Loops file the model actually used: the filtered / annotated BEDPE when the motif step ran, else the input."""
        if self.input_file:
            return str(self.input_file)
        ann = self.dirs.data / "loops_annotated.bedpe"
        return str(ann) if ann.is_file() else self.cfg.interaction_file

    @property
    def hic(self):
        """(map on the bead grid with NaN = masked, info) of the optional --hic file; None without one."""
        if self._hic is None and self.cfg.hic_file:
            from preproc.hic import hic_for_run
            self._hic = hic_for_run(self.cfg, self.geom, self.dirs.data / "metadata")
        return self._hic

    @property
    def mc_ens(self):
        """MC ensemble for maps, MD and structure checks: all chains pooled (replica 0 when there is one chain)."""
        if self._pool is None or self._pool[0] is not self.mc:
            self._pool = (self.mc, self.mc.pooled())
        return self._pool[1]


# ---- Chain the stages; each one computes (or loads a cached result) and describes it ----

log = get_logger(__name__)
TITLES = {"preproc": "Reading the input and describing it", "optimize": "Optimizing the number of LEFs, folding, binding and temperature (Nelder-Mead)", "mc": "Monte Carlo loop extrusion", "md": "3D molecular dynamics",
          "structure": "3D structure analysis and viewer", "analysis": "Validation against experiment and map features",
          "knots": "Linking numbers of loops", "report": "Writing report.html"}
STAGES = [stage_preproc, stage_optimize, stage_mc, stage_md, stage_structure, stage_analysis, stage_knots, stage_report]


def make_context(cfg, reuse_latest=False) -> RunContext:
    """Create the run folder, logging and the shared state."""
    set_quiet(cfg.quiet)
    print_banner()
    if cfg.seed is None:                                           # random run: draw a seed and record it, so the run can be repeated
        cfg.seed = random.randrange(2 ** 31 - 1000)
        cfg.source["seed"] = "random"
    np.random.seed(cfg.seed)                                       # numpy state of the main process (MD start structure, forces)
    geom = build_geometry(cfg)
    dirs = create_run_dirs(cfg.out_path, geom, cfg.run_number, reuse_latest)
    setup_logging(cfg.verbosity, dirs.log_file)
    REGISTRY.reset(dirs.figures)
    log.step("Run folder: %s", dirs.root)
    log.step("Region %s, %d threads, %d replica(s), seed %d", geom.label, apply_threads(cfg), cfg.replicas, cfg.seed)
    for note in cfg.notes:
        log.warning(note)
    write_ini(dirs.config, cfg.as_dict(), cfg.source)
    return RunContext(cfg, dirs, geom)


def _run_stage(ctx, st, compute: bool, k: int = 0, n: int = 0):
    """Compute the stage (or load its cache), then describe it."""
    path = ctx.dirs.data / st.CACHE if st.CACHE else None
    title = TITLES.get(st.NAME, st.NAME)
    if not compute and path is not None and path.exists():
        log.step("[%d/%d] %s: loading the saved result (%s)", k, n, title, path.name)
        obj = st.load(path)
    else:
        log.step("[%d/%d] %s", k, n, title)
        with Timer() as t:
            obj = st.compute(ctx)
        REGISTRY.timings.append((st.NAME, fmt_duration(t.seconds)))
        log.info("%s finished in %s", title, fmt_duration(t.seconds))
        if path is not None and obj is not None:
            (st.save(obj, path) if hasattr(st, "save") else obj.save(path))
    setattr(ctx, st.ATTR, obj)
    if obj is not None:
        if st.NAME != "report":
            log.info("Preparing results: tables and figures for '%s'", st.NAME)
        st.describe(ctx)


def run_pipeline(cfg, only=None) -> RunContext:
    """Run all enabled stages; with `only`, earlier stages reuse cached results."""
    ctx = make_context(cfg, reuse_latest=only is not None)
    names = [s.NAME for s in STAGES]
    last = names.index(only) if only else len(STAGES) - 1
    active = [(i, st) for i, st in enumerate(STAGES[:last + 1]) if st.enabled(cfg)]
    for i, st in enumerate(STAGES[:last + 1]):
        if not st.enabled(cfg):
            log.info("Stage '%s' disabled", st.NAME)
    for k, (i, st) in enumerate(active, 1):
        _run_stage(ctx, st, compute=(only is None or i == last or st is stage_report), k=k, n=len(active))
    log.step("All done. Results are in %s", ctx.dirs.root)
    return ctx


def run_standalone(stage_name, argv=None):
    """Entry for `python stage_xxx.py -c config.ini`."""
    run_pipeline(load_config(argv), only=stage_name)
