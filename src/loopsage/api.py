"""Python interface: independent functions, one per step (inputs, optimizer, Monte Carlo, MD, results, plots).

Each function takes a `Config` (see `config`) and the objects made by the previous step, and returns plain Python objects. Nothing is
written to disk unless a function is given a folder. `run` and `main` run the whole `loopsage` pipeline like the command.
"""
import atexit
import copy
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from settings.loader import Config, load_config, make_config


def parameters(search: str = None) -> dict:
    """All parameters as {name: (default, help)}; `search` keeps those whose name or help contains the text."""
    from settings.schema import PARAMS
    text = (search or "").lower()
    return {p.name: (p.default, p.help) for p in PARAMS if text in p.name.lower() or text in p.help.lower()}


def config(config=None, region=None, **params) -> Config:
    """A validated Config: `config` is an .ini path, a Config or a dict; `params` are parameters by name (they win)."""
    return make_config(config, region=region, **params)


def geometry(cfg: Config):
    """Region and bead grid of the configuration: chrom, start, end, n_beads, bp_per_bead, bead_of(bp), ..."""
    from preproc.loader import build_geometry
    return build_geometry(cfg)


def _tmp() -> Path:
    path = Path(tempfile.mkdtemp(prefix="loopsage_"))
    atexit.register(shutil.rmtree, path, True)
    return path


def _is_path(x) -> bool:
    return isinstance(x, (str, Path))


def read_inputs(cfg: Config, geom=None, loops=None, sites=None, compartments=None, tracks=None, track_names=None, lef_track=None):
    """Read the model inputs onto the bead grid and return `InputData`.

    Files are given as paths: `loops` (.bedpe), `sites` (.bed / .narrowPeak), `compartments` (.bw or .bed), `tracks` (a .bw path or a list
    of them), `lef_track` (.bw). Without `loops` and `sites` the config key `interaction_file` is used. Arrays work too (synthetic data):
    `loops=(left, right, weight)` in bp, `sites=dict(position=..., weight=..., reverse=...)`, per-bead arrays for the other inputs.
    """
    from preproc.arrays import patch_inputs, write_loops_bedpe, write_sites_bed
    from preproc.loader import load_inputs
    geom = geom or geometry(cfg)
    c, arrays, tmp = copy.copy(cfg), {}, _tmp()
    for key, given in (("interaction_file", loops), ("interaction_file", sites), ("lef_track_file", lef_track)):
        if _is_path(given):
            setattr(c, key, str(given))
    if _is_path(compartments):
        setattr(c, "comp_bw_file" if Path(compartments).suffix.lower() in (".bw", ".bigwig") else "comp_bed_file", str(compartments))
    if tracks is not None and (_is_path(tracks) or all(_is_path(t) for t in tracks)):
        c.bw_files = [str(t) for t in ([tracks] if _is_path(tracks) else tracks)]
    for name, writer, ext, spec, keys in (("loops", write_loops_bedpe, "bedpe", loops, ("left", "right", "weight")),
                                          ("sites", write_sites_bed, "bed", sites, ("position", "weight", "reverse"))):
        if spec is not None and not _is_path(spec):
            if loops is not None and sites is not None:
                raise ValueError("give loops or sites, not both")
            spec = dict(zip(keys, spec)) if isinstance(spec, (tuple, list)) else spec
            writer(tmp / f"{name}.{ext}", geom, **spec)
            c.interaction_file = str(tmp / f"{name}.{ext}")
    if not c.interaction_file:
        raise ValueError("give the loops: read_inputs(cfg, geom, loops='file.bedpe')")
    inputs = load_inputs(c, geom, c.interaction_file)
    per_bead = dict(tracks=None if _is_path(tracks) else tracks, lef=None if _is_path(lef_track) else lef_track,
                    comp=None if _is_path(compartments) else compartments)
    if any(v is not None for v in per_bead.values()):
        patch_inputs(inputs, geom, per_bead["tracks"], track_names, per_bead["lef"], per_bead["comp"],
                     max(1.5, geom.n_beads / 300) if cfg.smoothing_input else 0.0)
    inputs.source_file = c.interaction_file                       # the optimizer reads the loops again from this file
    return inputs


def read_hic(cfg: Config, geom=None, path=None):
    """Hi-C map of the region on the bead grid, as (matrix, info). `path` is a .hic file (default: config key `hic_file`)."""
    from preproc.hic import hic_for_run
    c = copy.copy(cfg)
    c.hic_file = str(path or cfg.hic_file or "")
    if not c.hic_file:
        raise ValueError("give a .hic file: read_hic(cfg, geom, path='map.hic')")
    return hic_for_run(c, geom or geometry(cfg), _tmp())


def optimize(cfg: Config, inputs, geom=None, hic=None):
    """Nelder-Mead search for the number of LEFs, folding, binding and temperature. `hic` (from `read_hic`) is the target,
    otherwise the loops of `inputs`. Returns an OptResult (`.best`, `.start`, `.trace`); `apply_optimum` puts it in a config."""
    from common.runtime import apply_threads
    from logger import set_quiet
    from pipeline import stage_optimize
    set_quiet(cfg.quiet)
    apply_threads(cfg)
    geom = geom or geometry(cfg)
    res = stage_optimize.compute(SimpleNamespace(cfg=cfg, geom=geom, hic=hic, inputs=inputs, loops_file=inputs.source_file))
    if res is None:
        raise ValueError("the optimizer needs a BEDPE loop file or a Hi-C map as the target")
    return res


def apply_optimum(cfg: Config, result) -> Config:
    """A copy of `cfg` with the optimized n_lef, folding_coeff, bind_coeff, t_init and t_final (and the Potts values when fitted)."""
    c, b = copy.copy(cfg), result.best
    c.n_lef, c.folding_coeff, c.bind_coeff, c.t_init, c.t_final = b["K"], b["f"], b["b"], b["t_init"], b["t_final"]
    if result.potts:
        c.epi_coeff, c.p_epigenetic = result.potts["best"]["epi"], result.potts["best"]["p"]
    return c


def run_mc(cfg: Config, inputs, geom=None):
    """Loop extrusion Monte Carlo: returns an MCResult (Ms, Ns, Es, Fs, Bs, Ks, Ts, ... per snapshot; `.chains` with replicas)."""
    from common.runtime import apply_threads
    from logger import set_quiet
    from monte_carlo.simulation import run_mc as _run_mc
    set_quiet(cfg.quiet)
    apply_threads(cfg)
    return _run_mc(cfg, inputs, geom or geometry(cfg), quiet=cfg.quiet)


def run_md(cfg: Config, mc, geom=None, folder=None):
    """3D molecular dynamics (OpenMM) from the Monte Carlo loops: returns an MDResult (coords, avg_map, rg, epot, ...).
    Without `folder` nothing is written; with a folder the structures and maps are saved there (as the command does)."""
    from common.runtime import apply_threads, OutputDirs
    from logger import set_quiet
    from molecular_dynamics.engine import run_md as _run_md
    set_quiet(cfg.quiet)
    apply_threads(cfg)
    c = copy.copy(cfg)
    root = Path(folder) if folder else _tmp()
    if not folder:
        c.save_ensemble = c.save_metadata = False
    dirs = OutputDirs(root)
    for d in (dirs.data / "metadata", dirs.structures):
        d.mkdir(parents=True, exist_ok=True)
    if c.seed is not None:
        np.random.seed(c.seed)
    return _run_md(c, geom or geometry(cfg), mc.pooled() if hasattr(mc, "pooled") else mc, dirs)


def run(config=None, region=None, **params):
    """The whole pipeline like the `loopsage` command (it writes the run folder); returns the RunContext."""
    from pipeline.runner import run_pipeline
    return run_pipeline(make_config(config, region=region, **params))


def main(argv=None):
    """The `loopsage` command from Python: `loopsage.main(["-c", "config.ini", "-vv"])`."""
    from pipeline.runner import run_pipeline
    return run_pipeline(load_config(argv))
