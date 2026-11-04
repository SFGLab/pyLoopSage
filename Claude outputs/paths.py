"""Output folder layout."""
from dataclasses import dataclass
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class OutputDirs:
    root: Path

    @property
    def data(self) -> Path: return self.root / "data"            # npy/npz arrays

    @property
    def figures(self) -> Path: return self.root / "figures"      # png + svg

    @property
    def structures(self) -> Path: return self.root / "structures"  # cif/psf/dcd

    @property
    def ensemble(self) -> Path: return self.structures / "ensemble"

    @property
    def tables(self) -> Path: return self.root / "tables"        # csv

    @property
    def report(self) -> Path: return self.root / "report.html"

    @classmethod
    def create(cls, path) -> "OutputDirs":
        root = Path(path).expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise IOError(f"'{root}' exists and is not a directory.")
        dirs = cls(root)
        for d in (dirs.data, dirs.figures, dirs.structures, dirs.tables):
            d.mkdir(parents=True, exist_ok=True)
        log.info("Output directory: %s", root)
        return dirs
