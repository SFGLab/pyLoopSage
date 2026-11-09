"""Sensitivity scans: many small sandboxed Monte Carlo (and optionally 3D) simulations over parameter choices, combined into one report."""
from loopsage.sensitivity.api import sweep
from loopsage.sensitivity.metrics import METRICS, metric
from loopsage.sensitivity.spec import ScanSpec, load_scan

__all__ = ["sweep", "load_scan", "ScanSpec", "metric", "METRICS"]
