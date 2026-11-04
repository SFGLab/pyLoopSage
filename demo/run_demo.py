"""Run a demo from Python (same as `python src/run.py -c demo/<name>.ini`).

    python demo/run_demo.py quick     # MC only, seconds
    python demo/run_demo.py full      # + 3D MD, validation, linking numbers
    python demo/run_demo.py epigenetic # MC + epigenetic model (dynamic coupling)
    python demo/run_demo.py speed     # MC with Maxwell-Boltzmann LEF speeds (speed and phase-space figures)
Options: --out DIR   results folder (default <project>/results)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

from make_demo_data import write_bedpe                  # noqa: E402
from pipeline.runner import run_pipeline                # noqa: E402
from settings.loader import load_config                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=["quick", "full", "epigenetic", "speed"], nargs="?", default="quick")
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", "--verbose", action="count", default=0)
    a = ap.parse_args()
    data = ROOT / "demo" / "data" / "demo_loops.bedpe"
    if not data.exists():
        write_bedpe(data)
    argv = ["-c", str(ROOT / "demo" / f"demo_{a.name}.ini"), "-f", str(data)] + ["-v"] * a.verbose
    if a.out:
        argv += ["-o", a.out]
    ctx = run_pipeline(load_config(argv))
    print(f"\nOpen the report: {ctx.dirs.report}")


if __name__ == "__main__":
    main()
