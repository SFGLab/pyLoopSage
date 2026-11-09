# Changelog

## 2.0.0rc3

- New `loopsage-scan` command and `ls.sweep()`: sensitivity scans of sandboxed loop-extrusion simulations (grid, one-at-a-time, random and Sobol designs), run in parallel with a cache, and combined into one `sensitivity_report.html` plus CSV tables.
- Report: importance map, Sobol indices, elasticities, heat maps, loop-size and per-bead profiles, convergence checks, and a section on phase transitions and bifurcations (fluctuations, critical slowing down, bifurcation diagrams, phase maps, Binder cumulant).
- Optional 3D mode with structure metrics and a topology mode with pooled link counts and crossing leaks versus the excluded-volume barrier.
- New reference page "Sensitivity scans"; examples `12_sensitivity_scan.ini` and `13_topology_scan.ini`.
- Packaging: `src` layout, `loopsage` and `loopsage-scan` entry points.

## 2.0.0rc2

- Previous pre-release.
