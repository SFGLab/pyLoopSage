"""Tables: shown in the terminal at -v and always stored for the report."""
from rich import box
from rich.table import Table as RichTable

from logger import console, get_verbosity
from common.units import fmt_num
from reporting.registry import REGISTRY


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
    from helpers.stats import describe
    d = describe(arr)
    return [name] + [fmt_num(d[k] * scale) for k in ("mean", "median", "min", "max")] + [unit]
