"""LoopSage as a Python library: one function per step.

    import loopsage as ls
    cfg = ls.config(region="chr1:77000000-95000000", n_beads=120, n_steps=20000)
    geom = ls.geometry(cfg)
    inputs = ls.read_inputs(cfg, geom, loops="loops.bedpe")      # files are given as paths
    mc = ls.run_mc(cfg, inputs, geom)                             # Monte Carlo
    ls.results.mean_loop_length(mc, geom)                         # results as arrays and tables
    ls.plots.energy_traces(mc)                                    # figures as matplotlib objects

    import loopsage.monte_carlo as mc                             # every sub-package is importable from here
    from loopsage.preproc.loader import load_inputs

Importing `loopsage` is cheap: the heavy modules load when first used.
"""
import importlib
import importlib.abc
import importlib.util
import sys

from version import __version__

# sub-packages and modules that can be imported as loopsage.<name>
_SUBMODULES = ("analysis", "common", "knots", "molecular_dynamics", "monte_carlo", "motif_finder", "optimizer", "pipeline",
               "preproc", "reporting", "settings", "validation", "viz", "viz3d", "logger")
# public names of the top-level API: name -> (module, attribute)
_API = {
    "config": ("loopsage.api", "config"),
    "parameters": ("loopsage.api", "parameters"),
    "geometry": ("loopsage.api", "geometry"),
    "read_inputs": ("loopsage.api", "read_inputs"),
    "read_hic": ("loopsage.api", "read_hic"),
    "optimize": ("loopsage.api", "optimize"),
    "apply_optimum": ("loopsage.api", "apply_optimum"),
    "run_mc": ("loopsage.api", "run_mc"),
    "run_md": ("loopsage.api", "run_md"),
    "run": ("loopsage.api", "run"),
    "main": ("loopsage.api", "main"),
    "Sandbox": ("loopsage.sandbox", "Sandbox"),
    "load_config": ("settings.loader", "load_config"),
    "Config": ("settings.loader", "Config"),
    "write_ini": ("settings.loader", "write_ini"),
}


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Make `loopsage.monte_carlo[.x]` the very same module object as `monte_carlo[.x]` (no second copy)."""

    def find_spec(self, name, path=None, target=None):
        parts = name.split(".")
        if parts[0] != "loopsage" or len(parts) < 2 or parts[1] not in _SUBMODULES:
            return None
        real = ".".join(parts[1:])
        try:
            if importlib.util.find_spec(real) is None:
                return None
        except (ImportError, ValueError):
            return None
        return importlib.util.spec_from_loader(name, self, is_package=importlib.util.find_spec(real).submodule_search_locations is not None)

    def create_module(self, spec):
        return importlib.import_module(spec.name.split(".", 1)[1])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())


def __getattr__(name):
    """Lazy attributes: ls.run_mc, ls.results, ls.plots, ls.load_config, ls.monte_carlo, ..."""
    if name in _API:
        mod, attr = _API[name]
        value = getattr(importlib.import_module(mod), attr)
    elif name in ("results", "plots"):
        value = importlib.import_module(f"loopsage.{name}")
    elif name in _SUBMODULES:
        value = importlib.import_module(f"loopsage.{name}")
    else:
        raise AttributeError(f"module 'loopsage' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_API) | set(_SUBMODULES))


__all__ = ["__version__", *_API, "results", "plots", *_SUBMODULES]
