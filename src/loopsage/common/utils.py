"""Wall-clock timer.

- Basic statistics used across stages.
- Tables: shown in the terminal at -v and always stored for the report.
"""
import time

import numpy as np
from rich import box
from rich.table import Table as RichTable
from scipy.stats import norm

from loopsage.common.genome import fmt_num
from loopsage.logger import console, get_verbosity
from loopsage.reporting.report import REGISTRY


# ---- Wall-clock timer ------------------------------------------------------

class Timer:
    """Context manager; read .seconds after exit."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


# ---- Basic statistics used across stages -----------------------------------

def describe(x) -> dict:
    """mean / median / min / max / std of an array (zeros if empty)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return dict(mean=0.0, median=0.0, min=0.0, max=0.0, std=0.0)
    return dict(mean=x.mean(), median=np.median(x), min=x.min(), max=x.max(), std=x.std())


def pearson_ci(r: float, n: int, alpha: float = 0.05):
    """95% CI of a Pearson r (Fisher z-transform)."""
    if n < 4 or abs(r) >= 1:
        return np.nan, np.nan
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(n - 3)
    crit = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def strength_label(r: float) -> str:
    """Plain-language label for |r| (Cohen-style thresholds)."""
    a = abs(r)
    for cut, name in ((0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"), (0.7, "strong")):
        if a < cut:
            return name
    return "very strong"


def normalize_signal(vec, log_transform: bool = True) -> np.ndarray:
    """Optional log1p (tames heavy tails), then z-score."""
    v = np.asarray(vec, dtype=float)
    if log_transform:
        v = np.log1p(np.clip(v, 0, None))
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def moving_average(y, window: int) -> np.ndarray:
    """Centred moving average that keeps the length (edges use fewer points)."""
    y = np.asarray(y, float)
    if window <= 1 or y.size < 3:
        return y
    k = np.ones(window)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")


# ---- Tables: shown in the terminal at -v and always stored for the report ----

def emit_table(title, columns, rows, caption="", section="overview", level=1):
    """Register a table for the report; print it if verbosity >= level."""
    rows = [[c if isinstance(c, str) else fmt_num(c) for c in r] for r in rows]
    REGISTRY.add_table(title, columns, rows, caption, section)
    if get_verbosity() < level:
        return
    t = RichTable(title=title, title_style="bold #6250d6", header_style="bold #6250d6",
                  box=box.ROUNDED, border_style="#b79cf0", caption=caption or None,
                  caption_style="dim italic")
    for c in columns:
        t.add_column(c)
    for r in rows:
        t.add_row(*r)
    console.print(t)


def describe_row(name, arr, unit, scale=1.0):
    """[name, mean, median, min, max, unit] with an optional unit conversion."""
    from loopsage.common.utils import describe
    d = describe(arr)
    return [name] + [fmt_num(d[k] * scale) for k in ("mean", "median", "min", "max")] + [unit]
