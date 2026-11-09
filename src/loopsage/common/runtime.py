"""Thread settings shared by the sampler, numba kernels and OpenMM.

- Stage progress bars (always shown).
- Output layout: <results>/<chrom>_<start>_<end>_<run>/{structures,visualizations,report,logs,data}.
"""
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from rich.live import Live
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from loopsage.logger import console, get_logger


# ---- Thread settings shared by the sampler, numba kernels and OpenMM -------

def n_threads(cfg) -> int:
    """Requested worker threads (0 or less = all cores)."""
    t = int(cfg.threads)
    return t if t > 0 else (os.cpu_count() or 1)


def apply_threads(cfg) -> int:
    """Limit numba's own thread pool to the requested size; returns the count."""
    import numba
    t = min(n_threads(cfg), numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(t)
    return t


# ---- Stage progress bars (always shown) ------------------------------------

@contextmanager
def stage_bar(desc: str, total: float):
    """Yield advance(n=1) that moves a purple progress bar."""
    cols = (TextColumn("[bold #6250d6]{task.description:<26}"),
            BarColumn(bar_width=None, complete_style="#8e6fd8", finished_style="#6250d6"),
            TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn())
    with Progress(*cols, console=console) as progress:
        task = progress.add_task(desc, total=total)
        yield lambda n=1: progress.advance(task, n)


# ---- Optimizer bar: rotating sage jokes and dancing pixels -----------------------------------------------

SAGE_LINES = (
    'The sage is looking for all possible future outcomes...',
    'Descending the loss landscape, one noisy step at a time...',
    'Gradient? Where we are going we do not need gradients...',
    'Nelder and Mead are pushing their simplex up the hill, then down...',
    'Local minimum found. Pretending it is the global one...',
    'Reflecting the worst vertex. It did not take it well...',
    'Contracting the simplex. It is getting cosy in here...',
    'Expanding the simplex. Hope is a strategy...',
    'The loss function is a landscape. We are very small hikers...',
    'Overfitting is just fitting with enthusiasm...',
    'Every evaluation is a coin flip with a loop map on it...',
    'Noisy loss detected. Blaming the Monte Carlo gods...',
    'Minimising regret, maximising correlation...',
    'The loss went down. Do not look at the noise, it will go up again...',
    'Searching 4 dimensions with a 5-cornered hat...',
    'Is it a valley or a ditch? Only the long run knows...',
    "Winner's curse loading... the best result is usually a bit lucky...",
    'Tuning knobs in the dark, listening for the correlation to rise...',
    'The optimum is somewhere near here. Probably. Maybe. Hopefully...',
    'Asking the contact map how wrong we are, politely...',
    'Loss = 1 - r + regret. Working on the regret...',
    'Hill climbing, but the hill is upside down...',
    'Convergence is not a destination, it is a tolerance...',
    'Temperature too high: chaos. Too low: stuck. Just right: tea time...',
    'Bias-variance trade-off: choosing which flavour of wrong we prefer...',
    'Random restarts would be a good idea. Sage is too proud...',
    'Saddle point ahead. Do not look down...',
    'Evaluating the objective. Objectively. Mostly...',
    'The simplex is shrinking, like my patience, but gentler...',
    'Taking a step in a promising direction. The direction disagrees...',
    'Rolling a fair die labelled K, f, b and T...',
    'Cohesin votes for a lower loss. CTCF abstains...',
    'Calibrating the oracle against the other oracle...',
    'The loss function has trust issues. So do we...',
    'Pareto front? No, just a very patient simplex...',
    'Standing on the shoulders of local minima...',
    'Tolerance set to optimism...',
    'Squinting at the loss curve until it looks like convergence...',
    'A watched loss never converges. Look away...',
    'Shaking the energy landscape to see what falls out...',
    'Bribing the LEFs with extra ATP...',
    'Teaching Nelder and Mead to dance...',
    'Counting loops, losing count, counting again...',
    'Reading the tea leaves of the contact map...',
    'The sage strokes its beard and re-runs the chain...',
)
_DANCERS = ("\u266a\u250f(\u25aa\u203f\u25aa)\u251b", "\u266b\u2517(\u25aa\u203f\u25aa)\u2513", "\u266a\u250f(\u25aa\u203f\u25aa)\u2513", "\u266b\u2517(\u25aa\u203f\u25aa)\u251b")
_PIXELS, _SHADES = "\u2596\u2597\u2598\u259d\u259e\u259a\u2584\u2580", ("#6a3de8", "#8e6fd8", "#b79cf0", "#e5307f", "#6250d6")


class _OptimizerView:
    """Two lines, redrawn about 8 times a second: bar + percent + time + loss, then a dancer, jumping pixels and a rotating joke."""

    def __init__(self, desc: str, total: float):
        self.desc, self.total, self.done, self.loss, self.best, self.t0 = desc, float(total), 0.0, None, None, time.time()
        self.joke, self.joke_t = random.choice(SAGE_LINES), self.t0

    def __rich_console__(self, console, options):
        tick = int(time.time() * 6)
        loss = Text("loss ...", style="#6b6485") if self.loss is None else Text.assemble(
            ("loss ", "#6b6485"), (f"{self.loss:.3f}", "bold #f08a24"), ("  best ", "#6b6485"), (f"{self.best:.3f}", "bold #1baf7a"))
        head = Table.grid(padding=(0, 1))
        head.add_row(Text(self.desc, style="bold #6250d6"), ProgressBar(total=self.total, completed=self.done, width=24, complete_style="#8e6fd8", finished_style="#6250d6"),
                     Text(f"{100 * self.done / self.total:3.0f}%", style="#6250d6"), Text(time.strftime("%M:%S", time.gmtime(time.time() - self.t0)), style="#6b6485"), loss)
        pixels = Text.assemble(*[(_PIXELS[(tick * (i + 3) + i * 5) % len(_PIXELS)], _SHADES[(tick + i) % len(_SHADES)]) for i in range(6)])
        if time.time() - self.joke_t > random.uniform(3.0, 6.0):             # a new random joke after a random pause, never the same twice in a row
            self.joke, self.joke_t = random.choice([x for x in SAGE_LINES if x != self.joke]), time.time()
        joke = Text(self.joke, style="italic #6b6485", no_wrap=True, overflow="ellipsis")
        yield head
        yield Text.assemble((_DANCERS[tick % 4] + " ", "bold #6250d6"), pixels, "  ", joke, no_wrap=True, overflow="ellipsis")


@contextmanager
def fun_bar(desc: str, total: float):
    """Optimizer progress (see _OptimizerView); yields tick(n=1, loss=..., best=...)."""
    view = _OptimizerView(desc, total)
    with Live(view, console=console, refresh_per_second=8, transient=False) as live:
        def prompt(text: str) -> str:
            """Ask a question under the bar: the live display is paused while the user answers."""
            live.stop()
            try:
                return console.input(escape(text))
            finally:
                live.start()

        def tick(n=1, loss=None, best=None):
            view.done += n
            if loss is not None:
                view.loss, view.best = loss, best
        tick.prompt, tick.notice = prompt, (lambda msg: console.print(f"[bold]{msg}[/bold]"))
        yield tick
        view.done = view.total


# ---- Output layout: <results>/<chrom>_<start>_<end>_<run>/{structures,visualizations,report,logs,data} ----

log = get_logger(__name__)


@dataclass(frozen=True)
class OutputDirs:
    root: Path                                           # one run folder

    @property
    def data(self) -> Path: return self.root / "data"                    # cached npz/npy arrays

    @property
    def figures(self) -> Path: return self.root / "visualizations"       # png + svg

    @property
    def viewer(self) -> Path: return self.figures / "3d"                  # 3d_viewer.html + videos

    @property
    def structures(self) -> Path: return self.root / "structures"        # cif / ensemble

    @property
    def ensemble(self) -> Path: return self.structures / "ensemble"

    @property
    def logs(self) -> Path: return self.root / "logs"

    @property
    def log_file(self) -> Path: return self.logs / "loopsage.log"

    @property
    def report(self) -> Path: return self.root / "report" / "report.html"

    @property
    def config(self) -> Path: return self.root / "config_resolved.ini"


def _runs(base: Path, prefix: str) -> dict:
    """{run_number: folder} of existing runs for one region."""
    found = {}
    for d in base.glob(f"{prefix}_*") if base.exists() else []:
        m = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", d.name)
        if m and d.is_dir():
            found[int(m.group(1))] = d
    return found


def create_run_dirs(base, geom, run=None, reuse_latest=False) -> OutputDirs:
    """Folder `<chrom>_<start>_<end>_<n>` under `base` (default ./results).

    run=None: next free number, or the latest existing one when reuse_latest.
    """
    base = Path(base).expanduser().resolve() if base else Path.cwd() / "results"
    if base.exists() and not base.is_dir():
        raise IOError(f"'{base}' exists and is not a directory.")
    prefix, runs = f"{geom.chrom}_{geom.start}_{geom.end}", None
    runs = _runs(base, prefix)
    if run is None:
        run = (max(runs) if runs else 1) if reuse_latest else (max(runs, default=0) + 1)
    dirs = OutputDirs(base / f"{prefix}_{run}")
    for d in (dirs.data, dirs.figures, dirs.structures, dirs.logs, dirs.report.parent, dirs.viewer):
        d.mkdir(parents=True, exist_ok=True)
    return dirs
