"""High-level Python interface: the `Simulation` class, `run`, `config` and `main`.

Every function here is a thin layer over the same stages that the command line runs.
"""
from pathlib import Path

import tempfile

from settings.loader import Config, load_config, make_config

# stage name -> Simulation method that runs it
_STAGES = ("preproc", "optimize", "mc", "md", "structure", "analysis", "knots", "report")


def parameters(search: str = None) -> dict:
    """All parameters as {name: (default, help)}; `search` keeps those whose name or help contains the text."""
    from settings.schema import PARAMS
    text = (search or "").lower()
    return {p.name: (p.default, p.help) for p in PARAMS if text in p.name.lower() or text in p.help.lower()}


def config(config=None, region=None, **params) -> Config:
    """A validated Config: `config` is an .ini path, a Config or a dict; `params` are parameters by name (they win)."""
    return make_config(config, region=region, **params)


class Simulation:
    """One LoopSage run, step by step or all at once. Each step first runs the steps it needs.

        sim = Simulation("config.ini", n_steps=20000)
        inputs = sim.preprocess()          # loops, anchors, tracks on the bead grid
        mc = sim.monte_carlo()             # loop extrusion
        sim.molecular_dynamics()           # 3D structures
        sim.run()                          # everything that is enabled, plus the report
    """

    def __init__(self, config=None, region=None, *, inputs=None, loops=None, sites=None, tracks=None, track_names=None,
                 lef_track=None, compartments=None, **params):
        self.cfg = make_config(config, region=region, **params)
        self.ctx = None
        self._done = set()
        self._arrays = dict(loops=loops, sites=sites, tracks=tracks, track_names=track_names, lef_track=lef_track, compartments=compartments)
        self._given = inputs                                   # ready InputData
        if inputs is None and loops is None and sites is None and not self.cfg.interaction_file:
            raise ValueError("give an input: interaction_file, loops=..., sites=... or inputs=...")

    # ---- state ----------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        """Folder of this run (created at the first step)."""
        return self._context().dirs.root

    @property
    def geometry(self):
        return self._context().geom

    def _context(self):
        if self.ctx is None:
            from pipeline.runner import make_context
            self.ctx = make_context(self.cfg)
        return self.ctx

    def __getattr__(self, name):
        """Results of finished steps: inputs, optimization, mc, md, structure, validation, features, links, report_path."""
        ctx = self.__dict__.get("ctx")
        if ctx is not None and not name.startswith("_") and hasattr(ctx, name):
            return getattr(ctx, name)
        raise AttributeError(name)

    # ---- steps ----------------------------------------------------------
    @property
    def _memory(self) -> bool:
        """True when the loops come from Python variables: only preprocessing, Monte Carlo and MD are available (they need no input file)."""
        return self._given is not None or self._arrays["loops"] is not None or self._arrays["sites"] is not None

    def _step(self, name):
        """Run one stage (and the enabled ones before it that did not run yet); returns its result."""
        from pipeline.runner import STAGES, _run_stage
        if self._memory and name not in ("preproc", "mc", "md"):
            raise ValueError(f"'{name}' needs the loops as a file (interaction_file); with loops given as Python variables use preprocess(), monte_carlo() and molecular_dynamics()")
        ctx = self._context()
        k = [s.NAME for s in STAGES].index(name)
        for i, st in enumerate(STAGES[:k + 1]):
            if st.NAME in self._done or (i < k and not (st.enabled(self.cfg) and st.NAME != "report")):
                continue
            if self._memory and st.NAME in ("optimize", "structure", "analysis", "knots"):
                continue                                         # these need the input file
            if self._memory:                                     # compute only: no tables, figures or report files
                setattr(ctx, st.ATTR, self._python_inputs(ctx.geom) if st.NAME == "preproc" else st.compute(ctx))
            else:
                _run_stage(ctx, st, compute=(i == k), k=1, n=1)
            self._done.add(st.NAME)
        return getattr(ctx, STAGES[k].ATTR)

    def _python_inputs(self, geom):
        if self._given is not None:
            return self._given
        from preproc.arrays import inputs_from_arrays
        return inputs_from_arrays(geom, **self._arrays)

    def preprocess(self):
        """Read loops and tracks (optional motif search): returns the InputData."""
        return self._step("preproc")

    def optimize(self):
        """Fit K, f, b and the temperature scale (Nelder-Mead); the best values are applied to the configuration."""
        return self._step("optimize")

    def monte_carlo(self):
        """Loop extrusion Monte Carlo: returns the MCResult."""
        return self._step("mc")

    def molecular_dynamics(self):
        """3D structures from the loop snapshots (OpenMM): returns the MDResult."""
        return self._step("md")

    def structure(self):
        """3D structure analysis and the interactive viewer."""
        return self._step("structure")

    def analyze(self):
        """Compare the simulated maps with the experiment: returns the ValidationResult."""
        return self._step("analysis")

    def links(self):
        """Linking numbers of loops in the 3D structures (needs topoly)."""
        return self._step("knots")

    def report(self):
        """Write report.html: returns its path."""
        return self._step("report")

    def run(self):
        """All enabled stages and the report, like the `loopsage` command. Returns self."""
        from pipeline.runner import run_pipeline
        if self._memory:
            raise ValueError("run() needs the loops as a file (interaction_file); with Python variables use preprocess(), monte_carlo() and molecular_dynamics()")
        if self.ctx is not None:
            raise RuntimeError("run() starts a fresh run; call it before the single steps, or make a new Simulation")
        self.ctx = run_pipeline(self.cfg)
        self._done = set(_STAGES)
        return self


def _inputs_and_geometry(cfg, inputs, arrays):
    from preproc.arrays import inputs_from_arrays
    from preproc.loader import build_geometry
    geom = build_geometry(cfg)
    if inputs is None:
        if arrays.get("loops") is None and arrays.get("sites") is None:
            from preproc.loader import load_inputs
            inputs = load_inputs(cfg, geom)                    # a file in the configuration
        else:
            inputs = inputs_from_arrays(geom, **arrays)
    return inputs, geom


def simulate(config=None, region=None, *, inputs=None, loops=None, sites=None, tracks=None, track_names=None,
             lef_track=None, compartments=None, **params):
    """Loop extrusion Monte Carlo entirely in memory: no run folder, no files, no figures. Returns the MCResult.

        mc = ls.simulate(loops=dict(left=l, right=r, weight=w), region="chr1:77000000-95000000", n_steps=20000, seed=1)
    """
    from monte_carlo.simulation import run_mc
    cfg = make_config(config, region=region, **params)
    inputs, geom = _inputs_and_geometry(cfg, inputs, dict(loops=loops, sites=sites, tracks=tracks, track_names=track_names,
                                                          lef_track=lef_track, compartments=compartments))
    return run_mc(cfg, inputs, geom, quiet=True)


def simulate_md(mc, config=None, region=None, **params):
    """3D molecular dynamics from the snapshots of a Monte Carlo result (`mc`). Returns the MDResult (coordinates, maps).

    OpenMM needs a few scratch files; they live in a temporary folder that is removed afterwards.
    """
    from common.runtime import OutputDirs
    from molecular_dynamics.engine import LoopMD
    from preproc.loader import build_geometry
    cfg = make_config(config, region=region, **params)
    geom = build_geometry(cfg)
    with tempfile.TemporaryDirectory(prefix="loopsage_md_") as tmp:
        dirs = OutputDirs(Path(tmp))
        for d in (dirs.data, dirs.structures, dirs.ensemble):
            d.mkdir(parents=True, exist_ok=True)
        engine = LoopMD(cfg, geom, mc.pooled(), dirs)
        engine.build()
        return engine.run()


def contact_map(mc, n_beads: int = None, **options):
    """Contact map (n_beads x n_beads) implied by the loops of a Monte Carlo result, all chains pooled."""
    from analysis.maps import loop_map
    return loop_map(mc.pooled(), n_beads or mc.meta["n_beads"], **options)


def run(config=None, region=None, **params) -> Simulation:
    """Run the whole pipeline: `loopsage.run("config.ini", n_steps=5000)`. Returns the finished Simulation."""
    return Simulation(config, region=region, **params).run()


def main(argv=None):
    """The `loopsage` command from Python: `loopsage.main(["-c", "config.ini", "-vv"])`."""
    from pipeline.runner import run_pipeline
    return run_pipeline(load_config(argv))
