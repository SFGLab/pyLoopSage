"""Compile the TikZ sources in tools/figures/ to SVG in markdown/assets/theory/ (needs pdflatex and pdftocairo)."""
import subprocess
import tempfile
from pathlib import Path

SRC = Path(__file__).parent / "figures"
OUT = Path(__file__).parents[1] / "markdown" / "assets" / "theory"


def build(tex: Path) -> None:
    """tex -> pdf (in a temp dir) -> svg."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdflatex", "-interaction=batchmode", f"-output-directory={tmp}", tex.name], cwd=SRC, check=True, capture_output=True)
        subprocess.run(["pdftocairo", "-svg", f"{tmp}/{tex.stem}.pdf", str(OUT / f"{tex.stem}.svg")], check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for t in sorted(SRC.glob("*.tex")):
        if t.name != "preamble.tex":
            build(t)
            print("built", t.stem)
