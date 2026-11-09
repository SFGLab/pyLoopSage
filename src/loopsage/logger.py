"""Verbosity-aware logging.

Console level by -v count: 0/1 -> main steps (log.step) and warnings (tables are printed
separately at -v), 2 -> info, 3 -> debug. The log file (if any) always records debug.
"""
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import install

install(show_locals=False)
console = Console()

ROOT = "loopsage"
STEP = 25                                         # between INFO and WARNING: milestones always shown
logging.addLevelName(STEP, "STEP")
logging.Logger.step = lambda self, msg, *a, **k: self.log(STEP, msg, *a, **k)
_LEVELS = {0: STEP, 1: STEP, 2: logging.INFO, 3: logging.DEBUG}
_verbosity = 0


class _Handler(RichHandler):
    """Rich handler that marks milestone messages with an arrow instead of a level name."""

    def get_level_text(self, record):
        return Text("▸", style="bold #6250d6") if record.levelno == STEP else super().get_level_text(record)


def get_verbosity() -> int:
    return _verbosity


def set_quiet(quiet: bool = True) -> None:
    """Silence all console output (banner, steps, tables, progress bars)."""
    console.quiet = bool(quiet)


def setup_logging(verbosity: int = 0, log_file=None) -> None:
    """Configure the 'loopsage' logger for the console and an optional file."""
    global _verbosity
    _verbosity = min(max(int(verbosity), 0), 3)
    root = logging.getLogger(ROOT)
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console_handler = _Handler(console=console, show_path=False, markup=False,
                                  show_time=_verbosity >= 2, rich_tracebacks=True)
    console_handler.setLevel(logging.CRITICAL + 10 if console.quiet else _LEVELS[_verbosity])
    root.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name.rsplit('.', 1)[-1]}")


def print_banner() -> None:
    """Show the LoopSage banner (lazy import avoids a cycle)."""
    from loopsage.common.banner import print_banner as _show
    _show()
