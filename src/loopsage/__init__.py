"""LoopSage as a Python library.

    import loopsage as ls
    sim = ls.Simulation("config.ini", n_steps=20000)      # or ls.Simulation(interaction_file=..., region="chr1:1-2000000")
    sim.run()                                             # the whole pipeline, same as the `loopsage` command

    import loopsage.monte_carlo as mc                     # every sub-package is importable from here
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
    "Simulation": ("loopsage.api", "Simulation"),
    "run": ("loopsage.api", "run"),
    "simulate": ("loopsage.api", "simulate"),
    "simulate_md": ("loopsage.api", "simulate_md"),
    "contact_map": ("loopsage.api", "contact_map"),
    "geometry": ("preproc.arrays", "geometry"),
    "inputs_from_arrays": ("preproc.arrays", "inputs_from_arrays"),
    "loops_from_arrays": ("preproc.arrays", "loops_from_arrays"),
    "sites_from_arrays": ("preproc.arrays", "sites_from_arrays"),
    "main": ("loopsage.api", "main"),
    "config": ("loopsage.api", "config"),
    "parameters": ("loopsage.api", "parameters"),
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
    """Lazy attributes: ls.Simulation, ls.run, ls.load_config, ls.monte_carlo, ..."""
    if name in _API:
        mod, attr = _API[name]
        value = getattr(importlib.import_module(mod), attr)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"loopsage.{name}")
    else:
        raise AttributeError(f"module 'loopsage' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_API) | set(_SUBMODULES))


__all__ = ["__version__", *_API, *_SUBMODULES]
