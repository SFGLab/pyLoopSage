"""Final stage: write report.html from everything collected."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logger import console
from reporting.report import build_report

NAME, ATTR, CACHE = "report", "report_path", None


def enabled(cfg) -> bool:
    return bool(cfg.report)


def compute(ctx):
    ctx.dirs.report.parent.mkdir(parents=True, exist_ok=True)
    ctx.dirs.report.write_text(build_report(ctx.cfg, f"Report - {ctx.geom.label}"), encoding="utf-8")
    return ctx.dirs.report


def describe(ctx):
    console.print(f"[bold #6250d6]Report:[/] {ctx.report_path}")


if __name__ == "__main__":
    from pipeline.runner import run_standalone
    run_standalone(NAME)
