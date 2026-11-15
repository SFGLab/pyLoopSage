# Changelog

## 2.0.0rc3

- MD forces use a cutoff (config keys `nonbonded_cutoff`, default 1 nm, and `comp_cutoff`, default 4 Gaussian widths; both 0 = all pairs): the cost per step grows about linearly instead of quadratically with the number of beads. The excluded volume is negligible beyond 1 nm and the compartment attraction keeps its long range.
- MD default `sim_step` (`--md-steps`) lowered from 5000 to 250 steps per frame (25 ps at the 100 fs time step): twenty times faster MD. In tests every new loop is closed by the end of its frame; raise `--md-steps` (for example to 1000) for more relaxation of the global shape.
- New `loopsage-scan` command and `ls.sweep()`: sensitivity scans of sandboxed loop-extrusion simulations (grid, one-at-a-time, random and Sobol designs), run in parallel with a cache, and combined into one `sensitivity_report.html` plus CSV tables.
- Report: importance map, Sobol indices, elasticities, heat maps, loop-size and per-bead profiles, convergence checks, and a section on phase transitions and bifurcations (fluctuations, critical slowing down, bifurcation diagrams, phase maps, Binder cumulant).
- Optional 3D mode with structure metrics and a topology mode with pooled link counts and crossing leaks versus the excluded-volume barrier.
- New reference page "Sensitivity scans"; examples `12_sensitivity_scan.ini` and `13_topology_scan.ini`.
- Packaging: `src` layout, `loopsage` and `loopsage-scan` entry points.

## 2.0.0rc2

- Previous pre-release.
