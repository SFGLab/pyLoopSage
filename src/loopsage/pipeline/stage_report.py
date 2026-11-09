"""Final stage: write report.html from everything collected."""

from loopsage.logger import console
from loopsage.reporting.report import build_report

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
    from loopsage.pipeline.runner import run_standalone
    run_standalone(NAME)
