"""Figures as matplotlib objects: `loopsage.plots.energy_traces(mc)` returns the Figure (or a list of them) to show or save.

Any figure function of `loopsage.viz` can be called this way by its name, e.g. `kymograph(mc, geom, inputs)`.
"""
import importlib
import inspect

import matplotlib.pyplot as plt

_MODULES = ("mc_figs", "map_figs", "input_figs", "optimizer_figs", "structure_figs", "validation_figs", "hic_figs", "nesting_figs", "speed_figs")


def _find(name):
    for mod in _MODULES:
        fn = getattr(importlib.import_module(f"viz.{mod}"), name, None)
        if inspect.isfunction(fn) and fn.__module__ == f"viz.{mod}":
            return fn
    raise AttributeError(f"no figure function '{name}' in loopsage.viz")


def _wrap(fn):
    def draw(*args, **kwargs):
        from reporting.report import REGISTRY
        old = REGISTRY.capture
        REGISTRY.capture = captured = []
        try:
            fn(*args, **kwargs)
        finally:
            REGISTRY.capture = old
        return captured[0] if len(captured) == 1 else (captured or None)
    draw.__name__, draw.__doc__ = fn.__name__, fn.__doc__
    return draw


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return _wrap(_find(name))


def show():
    """Show the open figures (plt.show)."""
    plt.show()
