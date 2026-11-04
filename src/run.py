"""LoopSage entry point: the whole pipeline from a config file or the command line.

    python run.py -c config.ini
    python run.py -f loops.bedpe -r chr1:80000000-90000000 --run-MD -vv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # lets VS Code run it directly

from pipeline.runner import run_pipeline
from settings.loader import load_config


def main(argv=None):
    run_pipeline(load_config(argv))


if __name__ == "__main__":
    main()
