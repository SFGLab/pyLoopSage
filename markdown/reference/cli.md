# Command-line interface

LoopSage is run with one command. Every option below can also be written in a [config file](config.md); the command line always wins.

```bash
loopsage [options]
```

!!! tip "Where options are read from"
    From weakest to strongest: built-in defaults, then the config file (`-c`), then command-line options, then `-s KEY=VALUE`.
    So a config file holds your usual settings and the command line changes one thing at a time.

## Examples

```bash
# Minimal run: a loop file and a region
loopsage -f loops.bedpe -r chr1:77000000-95000000

# Everything from a config file, but without the 3D stage and with info messages
loopsage -c config.ini --no-run-MD -vv

# Read CTCF orientation from the genome, keep loops with at least 3 PET counts, 4 threads
loopsage -f loops.bedpe -r chr1:77000000-95000000 --motif-genome hg38.fa --min-pet 3 -t 4

# Set any config key that has no option of its own
loopsage -c config.ini -s epi_coeff=0.5 -s bind_coeff=2
```

## General options

| Option | What it does |
|---|---|
| `-h`, `--help` | Show the help text and exit. |
| `-c FILE`, `--config FILE` | Read settings from a sectioned `.ini` file (see [Config files](config.md)). |
| `-r CHR:START-END`, `--region CHR:START-END` | Genomic region in base pairs, for example `chr1:75000000-95000000`. |
| `-v`, `-vv`, `-vvv` | Console detail. None: main steps and progress bars. `-v`: plus tables. `-vv`: plus info messages. `-vvv`: plus debug. |
| `-s KEY=VALUE`, `--set KEY=VALUE` | Set any config key from the command line. Can be repeated. |

## Options by topic

Switches that are on or off come in pairs: `--run-MD` turns a step on and `--no-run-MD` turns it off. Lists in the *Type* column show the unit in brackets when the value has one.

### Input data and genomic region

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `-f`, `--file` | text | *(empty)* | Loops/peaks file: .bedpe, .bed or .narrowPeak (required) |
| `--hic` | text | *(empty)* | Optional Hi-C map (.hic, read with hic-straw): reference for the validation and, with --optimize, the target of the optimizer |
| `--hic-norm` | text | `KR` | Hi-C normalisation to read (falls back to SCALE, VC_SQRT, VC, NONE when missing) |
| `--hic-resolution` | integer (bp) | `0` | Hi-C bin size to read (0 = automatic, closest to the bead size) |
| `--genome` | text | `hg38` | Genome assembly for chromosome sizes |
| `-n`, `--n-beads` | integer (beads) | *(empty)* | Number of simulation beads (empty = region / 2 kb) |
| `--lef-track` | text | *(empty)* | BigWig of LEF loading preference (cohesin/condensin ChIP) |
| `--min-pet` | integer (PET) | `0` | BEDPE: keep only loops with at least this many PET counts / score (0 = keep all) |
| `--motif-genome` | text | *(empty)* | BEDPE: reference genome FASTA; enables the CTCF motif search that sets anchor orientations |
| `--motif-pfm` | text | *(empty)* | Motif PFM file (empty = bundled JASPAR CTCF MA0139.1) |
| `--motif-pvalue` | number (p-value) | `1e-05` | Motif hit cutoff: chance of a hit in random sequence (smaller = stricter) |

### Output and reproducibility

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `-o`, `--out` | text | *(empty)* | Results folder (empty = ./results in the current folder); each run gets its own subfolder |
| `--run` | integer | *(empty)* | Run number in the folder name (empty = next free; standalone stages reuse the latest) |
| `-q`, `--quiet` / `--no-quiet` | true / false | `False` | Print nothing to the console (the log file and the report are still written) |
| `--seed` | integer | *(empty)* | Random seed (empty = a random one, saved in the resolved config); the same seed repeats the run exactly, MD included only with --threads 1 |
| `--report` / `--no-report` | true / false | `True` | Write report.html |
| `--figures` / `--no-figures` | true / false | `True` | Draw and save the figures (off: tables and data only, much faster for big or repeated runs) |
| `--metadata` / `--no-metadata` | true / false | `True` | Save map matrices (MC and MD heatmaps, distances, strengths) as .npy in data/metadata |
| `--save-ensemble` / `--no-save-ensemble` | true / false | `True` | Save the 3D structures: every MD frame as .cif, plus the start structure, topology and trajectory (structures/) |

### Parallelism and replicas

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `-t`, `--threads` | integer (threads) | `1` | Worker threads for the numba sampler, OpenMM CPU and numba kernels (0 = all cores) |
| `--n-replicas` | integer (chains) | `1` | Independent Monte Carlo chains (seed, seed+1, ...) run in parallel; the ensemble is pooled |

### Monte Carlo

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `--mc-steps` | integer (MC steps) | `20000` | Total Monte Carlo steps |
| `--sample-every` | integer (MC steps) | `200` | Save one snapshot every this many steps |
| `--burnin` | integer (MC steps) | `1000` | Burn-in period discarded from statistics |
| `--t-init` | number (a.u.) | `3.0` | Initial temperature |
| `--t-final` | number (a.u.) | `1.0` | Final temperature (Annealing only) |
| `--method` | text | `Annealing` | Sampling method |
| `--n-lef` | integer (LEFs) | *(empty)* | Loop extrusion factors (empty = N_anchors / 2) |
| `--lef-rw` / `--no-lef-rw` | true / false | `True` | LEF legs slide as a random walk (True) or only extrude outward (False) |
| `--lef-speed` | text | `constant` | Initial LEF speeds: constant (all equal), maxwell (Maxwell-Boltzmann), lognormal, or two_state (slow and fast LEFs); mean speed is always 1 |
| `--lef-speed-spread` | number | `0.5` | lognormal only: width (sigma of ln speed) |
| `--lef-fast-fraction` | number | `0.2` | two_state only: fraction of fast LEFs |
| `--lef-fast-ratio` | number | `5.0` | two_state only: speed of fast LEFs relative to slow ones |

### Energy model coefficients

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `--binding-mode` | text | `preferential` | Anchor binding mode: preferential = LEFs are attracted to the anchor profiles (non-negative), loop-free regions cost nothing; contrastive = in addition, loops are penalised for spanning loop-free regions and rewarded for spanning regions the input loops cover (needs loops) |

### Molecular dynamics

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `--run-MD` / `--no-run-MD` | true / false | `True` | Run the 3D molecular dynamics stage |
| `--platform` | text | `auto` | OpenMM platform: auto (try CUDA, then OpenCL, then parallel CPU), CUDA, OpenCL, CPU or Reference |
| `--init-structure` | text | `rw` | Initial polymer shape |
| `--step-size` | number (fs) | `100.0` | Integrator time step |
| `--md-steps` | integer (MD steps) | `5000` | MD steps per MC snapshot (per frame) |
| `--temperature` | number (K) | `310.0` | Simulation temperature |

### 3D structure analysis and rendering

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `--render-3d` / `--no-render-3d` | true / false | `True` | Write 3d_viewer.html and videos of the 3D trajectory |

### Analysis

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `--track-smooth` | number (beads) | `7.0` | Gaussian smoothing of the Hi-C insulation score and of the A/B track (Hi-C eigenvector or compartment BigWig) used as input and for validation (0 = off) |

### Parameter optimizer

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `--optimize` / `--no-optimize` | true / false | `False` | Before the run, fit the number of LEFs, folding and binding coefficients with Nelder-Mead on short Monte Carlo runs (n_lef of the first family, f, b and the temperature schedule), then run with them |
| `--opt-evals` | integer (evaluations) | `60` | Maximum number of short Monte Carlo evaluations of the optimizer |
| `--opt-early-stop` / `--no-opt-early-stop` | true / false | `True` | Stop the optimizer early when the loss no longer improves |
| `--opt-interactive` / `--no-opt-interactive` | true / false | `True` | Pressing Enter during the optimizer shows the best parameters so far and asks whether to keep them and stop (only in a terminal) |
| `--opt-patience` | integer (evaluations) | `20` | Early stopping: evaluations without a real improvement (see opt_min_gain) before the optimizer stops |
| `--opt-steps` | integer (MC steps) | `4000` | Monte Carlo steps of one optimizer evaluation (each step is a sweep over all LEFs; the first quarter is burn-in, 200 snapshots; the final candidate check runs 3x longer) |
| `--opt-max-beads` | integer (beads) | `1500` | Largest region the optimizer simulates: bigger regions are optimized on the window of this many beads with most loops, and the number of LEFs is scaled back to the full region |
| `--target-smooth` | number (beads) | `0.8` | Smoothing of the optimizer target built from the input loops, so that the sparse loops look like a Hi-C map (0 = none) |
| `--hic-denoise` | number (beads) | `1.0` | Denoising of the Hi-C target of the optimizer, to bring out the TAD pattern (0 = none); validation always uses the raw map |

### Topology

| Option | Type (unit) | Default | What it does |
|---|---|---|---|
| `--knots` / `--no-knots` | true / false | `False` | Optional linking analysis of loops in 3D (slow; off by default) |

## Output

Every run gets its own folder, `results/<chrom>_<start>_<end>_<run>/`, with the report in `report/report.html`, the figures in `visualizations/`, the 3D structures in `structures/`, and the numerical results in `data/` (matrices in `data/metadata/` as `.npy` files).
Use `-o` to choose another results folder and `--run N` to fix the run number.

**Everything is saved by default**, from the command line and from a config file. Each kind of output has its own switch to turn it off (`--no-...`, or `... = False` in the config file):

| Output | Where | Switch to turn it off |
|---|---|---|
| HTML report | `report/report.html` | `--no-report` |
| Figures (PNG and SVG) | `visualizations/` | `--no-figures` |
| Matrices as `.npy` (contact maps, distances, strengths) | `data/metadata/` | `--no-metadata` |
| 3D structures: every MD frame as `.cif`, start structure, topology, trajectory | `structures/` | `--no-save-ensemble` |
| 3D viewer and videos | `visualizations/3d/` | `--no-render-3d` |

The results of the stages (`data/*.npz`), the log and `config_resolved.ini` are always written, because the stages and the repeatability of a run depend on them. From [Python](python.md), the step functions (`ls.run_mc`, `ls.run_md`, ...) write nothing; `ls.run` writes the same run folder as the command.

## Run only some stages

The pipeline has seven stages: reading the input, Monte Carlo, molecular dynamics, 3D structure analysis, validation, linking analysis and the report.
Each stage can also be started alone with the same options; it re-uses the saved result of the earlier stages from the latest run.

```bash
python -m loopsage.pipeline.stage_mc -c config.ini      # Monte Carlo (and the input before it)
python -m loopsage.pipeline.stage_md -c config.ini      # up to the 3D molecular dynamics
```
