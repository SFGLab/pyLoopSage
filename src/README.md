# LoopSage (refactored)

Run the whole pipeline:

    python src/run.py -c config_example.ini
    python src/run.py -f loops.bedpe -r chr1:80000000-90000000 --run-MD -v

Verbosity: no flag = progress bars only; `-v` = result tables; `-vv` = + info messages; `-vvv` = debug.
Results go to `results/` in the project root (git-ignored), one folder per run, named `{chrom}_{start}_{end}_{run}`:

    results/chr1_77000000_95000000_1/
        structures/      3D structures (.cif)
        visualizations/  every figure as png + svg
        report/          report.html
        logs/            loopsage.log (always debug level)
        data/            cached stage outputs (npz / npy)
        config_resolved.ini

Use `--out DIR` for another results folder and `--run N` to pick a run number.

Layout (each file is small and documented):

| Folder | Purpose |
|---|---|
| `settings/` | parameter schema, CLI + ini parsing (defaults < ini < CLI) |
| `motif_finder/` | optional CTCF motif search: indexed FASTA reader, PFM scan, anchor orientation probabilities |
| `preproc/` | read bedpe / bed / narrowPeak / BigWig into bead resolution |
| `monte_carlo/` | LEF sampler (`energy/` holds each energy term) |
| `molecular_dynamics/` | OpenMM polymer model, forces, contact maps |
| `analysis/`, `validation/` | diagnostics, 3D shape metrics, map features (PC1, insulation, P(s)), comparison with experiment |
| `knots/` | optional (`--knots`, off by default): Gauss linking of loops in 3D, closest-approach, link maps, 3D snapshots |
| `viz3d/` | pyvista videos (tube by default, beads optional) and the interactive `3d_viewer.html` (optional deps: `pyvista`, `imageio-ffmpeg`) |
| `viz/` | figures (white background, at most 3 panels each) |
| `reporting/`, `html/` | report builder; `html/` holds the CSS and page template |
| `pipeline/` | stages (`stage_*.py`) and the runner |

Parallelism: `--threads N` (`-t`, 0 = all cores) sets worker threads for the sampler, OpenMM CPU and numba kernels; `--n-replicas R` runs R independent Monte Carlo chains (seeds `seed`, `seed+1`, ...) in parallel threads (the numba sampler releases the GIL). Results do not depend on the thread count. Maps, 3D structures and diagnostics use all chains pooled, and the report adds a replica-agreement figure (R-hat, occupancy correlation). In `config.ini`: `[parallel]` with `threads` and `replicas`.

Any stage runs alone and reuses cached results from `data/`, e.g.
`python src/pipeline/stage_md.py -c config.ini`.

## Epigenetic model (optional)

Off by default (`epi_coeff = 0`: pure loop extrusion). Set `epi_coeff > 0` in `config.ini` to give every bead an epigenetic state
(`n_epi_states`) that prefers the state of its neighbours. With `epigenetic_mode = dynamic` (default when enabled; config.ini only) the
coupling `J` is 1 on the backbone and on the current loops and 0 elsewhere, so it follows the LEFs; `static` uses a fixed map from the
loop data. Energy changes are computed incrementally for both state changes and LEF moves. Try `python demo/run_demo.py epigenetic`.
