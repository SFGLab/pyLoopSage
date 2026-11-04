"""Thread settings shared by the sampler, numba kernels and OpenMM.

- Stage progress bars (always shown).
- Output layout: <results>/<chrom>_<start>_<end>_<run>/{structures,visualizations,report,logs,data}.
"""
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from logger import console, get_logger


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


# ---- Output layout: <results>/<chrom>_<start>_<end>_<run>/{structures,visualizations,report,logs,data} ----

log = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]       # folder that contains src/
DEFAULT_RESULTS = PROJECT_ROOT / "results"


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
    """Folder `<chrom>_<start>_<end>_<n>` under `base` (default <project>/results).

    run=None: next free number, or the latest existing one when reuse_latest.
    """
    base = Path(base).expanduser().resolve() if base else DEFAULT_RESULTS
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
