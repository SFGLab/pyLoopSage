# Changelog

## Unreleased

- Validation: the per-position profiles (loop strength, insulation score, boundary strength, compartments) are smoothed with one Gaussian width, `track_smooth` (the same as the Hi-C insulation and A/B tracks); correlations, the summary and the figures use the smoothed profiles, the raw correlation stays as a `, raw` row. `corr_smooth` is removed (ignored with a note).
- Validation of the contact decay P(s): the experimental curve now comes from the loop map drawn like the simulated one, and experiment, loop ensemble (MC) and 3D (MD) are all normalised to probabilities (sum 1), so the curves share one scale in the figure; the correlation is taken on log P(s). The per-loop observed/expected values of both sides are rescaled to mean 1.
- `--out` is the output folder itself: it is created with its parents if missing and used exactly as given (`--out results` writes into `results/`, `--out results/chr1_134_489` into that subfolder); the region is no longer part of the layout, only of the report. Without `--out` the folder is `results/<chrom>_<start>_<end>`. Earlier: the run folder had no run number; the `run_number` option and `--run` are removed (old config files only get an 'ignored' note). Running the same region again into the same `--out` folder overwrites it (a warning is logged); use another `--out` to keep both.
- Replicas are fully independent: each has its own Monte Carlo chain (own seed, random initial loops) and its own 3D run (own start structure, loops and random stream). With one replica nothing changes in the output; with two or more, per-replica metadata, structures and figures go to `replica_<k>` subfolders and the report adds the mean and standard deviation of the contact and distance maps across replicas (`replica_maps`, `md_replica_maps`, `md_replica_distances`, `md_replicas`).
- Replica 0 uses `--seed`; every other replica gets its own independent seed derived from it (separate streams for its Monte Carlo chain and for its 3D start structure and MD noise). The seeds are listed in the report, and replica `k` equals a single run started with the seed listed for it.
- With several replicas the cached results are `mc_r0.npz`, `mc_r1.npz`, ... and `md_r0.npz`, ... (one replica: `mc.npz`, `md.npz`; `MCResult.load` / `MDResult.load` accept `mc.npz` in both cases); the Python `run_md` returns an `MDResult` with `.chains` and `.pooled()`.

## 2.0.1

- Dependencies state numpy 2 (`numpy>=2.0`, `numba>=0.60`), the version LoopSage is developed and tested on.
- Installation fixes. The old `OpenMM-cuda` dependency pinned OpenMM 8.1 and made pip downgrade it; it is removed. GPU builds are now extras: `pip install "pyLoopSage[cuda12]"` or `[cuda13]`.
- Only the packages the code imports are required (removed `mdtraj`, `seaborn`, `scikit-learn`, `statsmodels`, `tqdm`, `typing-extensions`, `pyvista[all]`). Optional features are extras: `[bigwig]`, `[hic]`, `[topology]`, `[all]`.
- `pyproject.toml` replaces the metadata in `setup.py`; works with pip (also in pyenv) and uv.

## 2.0.0

First stable release of LoopSage 2 (includes everything from the release candidates).

- MD forces use a cutoff (config keys `nonbonded_cutoff`, default 1 nm, and `comp_cutoff`, default 4 Gaussian widths; both 0 = all pairs): the cost per step grows about linearly instead of quadratically with the number of beads. The excluded volume is negligible beyond 1 nm and the compartment attraction keeps its long range.
- MD default `sim_step` (`--md-steps`) lowered from 5000 to 250 steps per frame (25 ps at the 100 fs time step): twenty times faster MD. In tests every new loop is closed by the end of its frame; raise `--md-steps` (for example to 1000) for more relaxation of the global shape.
- New `loopsage-scan` command and `ls.sweep()`: sensitivity scans of sandboxed loop-extrusion simulations (grid, one-at-a-time, random and Sobol designs), run in parallel with a cache, and combined into one `sensitivity_report.html` plus CSV tables.
- Report: importance map, Sobol indices, elasticities, heat maps, loop-size and per-bead profiles, convergence checks, and a section on phase transitions and bifurcations (fluctuations, critical slowing down, bifurcation diagrams, phase maps, Binder cumulant).
- Optional 3D mode with structure metrics and a topology mode with pooled link counts and crossing leaks versus the excluded-volume barrier.
- New reference page "Sensitivity scans"; examples `12_sensitivity_scan.ini` and `13_topology_scan.ini`.
- Packaging: `src` layout, `loopsage` and `loopsage-scan` entry points.

## 2.0.0rc2

- Pre-release.
