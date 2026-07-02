import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Pretty tracebacks
install(show_locals=False)

console = Console()


def get_logger(name: str = "pyLoopSage",
               level: int = logging.INFO,
               log_file: str | None = None) -> logging.Logger:
    """
    Create a colorful logger for console and optional log file.
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # ---------- Console ----------
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_time=True,
    )

    console_handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    logger.addHandler(console_handler)

    # ---------- File ----------
    if log_file is not None:

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)

        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] "
                "%(levelname)-8s "
                "%(name)s :: "
                "%(message)s",
                "%Y-%m-%d %H:%M:%S"
            )
        )

        logger.addHandler(file_handler)

    logger.propagate = False

    return logger