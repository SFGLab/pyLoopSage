"""Console output: milestone messages at default verbosity, clean progress bars, quote only under the banner."""
import pytest

from common.banner import print_banner, QUOTES
from common.runtime import stage_bar
from logger import console, get_logger, setup_logging
from pipeline.runner import run_pipeline
from settings.loader import load_config


@pytest.fixture(autouse=True)
def _restore_logging():
    yield
    setup_logging(0)


def _shown(verbosity):
    setup_logging(verbosity)
    log = get_logger("test_console")
    with console.capture() as cap:
        log.step("milestone"); log.info("detail"); log.debug("trace"); log.warning("careful")
    return cap.get()


def test_default_verbosity_shows_milestones_and_warnings_only():
    out = _shown(0)
    assert "milestone" in out and "careful" in out and "detail" not in out and "trace" not in out
    assert "▸" in out                                              # milestones carry the arrow marker


def test_verbosity_levels_add_detail():
    assert "detail" in _shown(2) and "trace" not in _shown(2)
    assert "trace" in _shown(3) and "milestone" in _shown(3)


def test_progress_bar_has_no_quote():
    with console.capture() as cap:
        with stage_bar("Testing bar", 10) as advance:
            for _ in range(10):
                advance()
    out = cap.get()
    assert "Testing bar" in out and "100%" in out
    assert not any(text[:25] in out for text, _ in QUOTES)


def test_banner_shows_one_rotating_quote():
    with console.capture() as cap:
        print_banner()
    out = cap.get()
    assert "LOOPSAGE" in out.upper() or "██" in out
    seen = {q for q in QUOTES if q[0][:20] in out.replace("\n", " ")}
    assert len(seen) <= 1
    picks = set()
    for _ in range(40):
        from common.banner import pick
        picks.add(pick())
    assert len(picks) > 1                                          # different quote on different runs


def test_run_logs_each_stage(fast_args):
    ctx = run_pipeline(load_config(fast_args))
    log = ctx.dirs.log_file.read_text()
    for msg in ("Run folder", "[1/", "Reading the input", "Sampling", "Monte Carlo loop extrusion finished", "Preparing results",
                "Writing report.html", "All done"):
        assert msg in log
