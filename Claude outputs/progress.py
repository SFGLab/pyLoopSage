"""Stage progress bars (always shown, regardless of verbosity)."""
from contextlib import contextmanager

from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from logger import console


@contextmanager
def stage_bar(desc: str, total: float):
    """Yield advance(n=1) that moves a purple progress bar."""
    cols = (TextColumn("[bold #6250d6]{task.description:<26}"),
            BarColumn(bar_width=None, complete_style="#8e6fd8", finished_style="#6250d6"),
            TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn())
    with Progress(*cols, console=console) as progress:
        task = progress.add_task(desc, total=total)
        yield lambda n=1: progress.advance(task, n)
