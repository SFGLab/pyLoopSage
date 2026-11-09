# Config files

A config file keeps all the settings of a run in one place, so a run can be repeated exactly.
Pass it with `-c`:

```bash
loopsage -c config.ini
```

`config_example.ini` in the repository root lists every key with its default and a one-line description. Copy it and change what you need.

## How to write one

A config file is a plain text file in the INI format: *sections* in square brackets, and one `key = value` per line.

```ini
[input]
interaction_file = /data/loops.bedpe   ; the loops to model
chrom = chr1
region_start = 77000000                ; in base pairs
region_end = 95000000

[monte_carlo]
n_steps = 20000
method = Annealing

[molecular_dynamics]
run_md = True
```

Rules:

- **Sections** group the keys by topic (tables below). A key must be written in its own section.
- **Comments** are whole lines starting with `#` or `;`. A comment may also follow a value on the same line, after `;`.
- **Empty values** (`seed =`) mean "not set": LoopSage then picks a sensible automatic value.
- **Booleans** are written `True` / `False` (also `yes` / `no`, `1` / `0`).
- **Lists** are comma-separated: `bw_files = a.bw, b.bw`.
- **Units** may follow a number (`step_size = 100 fs`); the unit of each key is given in the tables.
- **Only the keys you change are needed.** Anything missing takes its default.
- **Every command-line option is also a config key**, but not the other way round: the config file has more (advanced) keys, marked *ini only* below.

!!! tip "Check what was used"
    Each run saves the full list of settings it used in `config_resolved.ini` inside the run folder, so any run can be reproduced from its own folder.

## Using the settings from Python

The same keys work in Python, written by name (`n_steps`, not the option `--mc-steps`):

```python
import loopsage as ls

cfg = ls.config("config.ini", n_steps=5000, run_md=False)    # a validated Config; keyword arguments win over the file
geom = ls.geometry(cfg)
inputs = ls.read_inputs(cfg, geom, loops="loops.bedpe")      # inputs as file paths
mc = ls.run_mc(cfg, inputs, geom)                            # one function per step, results as Python objects
```

Values are converted exactly as in a config file, a wrong key or value raises `ValueError`, and `cfg.as_dict()` lists every setting. The full description (inputs, optimizer, Monte Carlo, molecular dynamics, results, figures, the sandbox) is on the page [Python interface](python.md).

## All keys

**Type** tells how to write the value. Under each key is the equivalent command-line option, or *ini only* when the key has none.

### Input data and genomic region

Section `[input]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `interaction_file`<br>`--file` | text | *(empty)* | Loops/peaks file: .bedpe, .bed or .narrowPeak (required) |
| `hic_file`<br>`--hic` | text | *(empty)* | Optional Hi-C map (.hic, read with hic-straw): reference for the validation and, with --optimize, the target of the optimizer |
| `hic_norm`<br>`--hic-norm` | text | `KR` | Hi-C normalisation to read (falls back to SCALE, VC_SQRT, VC, NONE when missing) |
| `hic_resolution`<br>`--hic-resolution` | integer (bp) | `0` | Hi-C bin size to read (0 = automatic, closest to the bead size) |
| `chrom`<br>*ini only* | text | *(empty)* | Chromosome, e.g. chr1 (CLI: use --region) |
| `region_start`<br>*ini only* | integer (bp) | *(empty)* | Region start (empty = chromosome start) |
| `region_end`<br>*ini only* | integer (bp) | *(empty)* | Region end (empty = chromosome end) |
| `genome`<br>`--genome` | text | `hg38` | Genome assembly for chromosome sizes |
| `n_beads`<br>`--n-beads` | integer (beads) | *(empty)* | Number of simulation beads (empty = region / 2 kb) |
| `smoothing_input`<br>*ini only* | true / false | `True` | Smooth the anchor/peak signal and the BigWig binding tracks with a Gaussian (sigma = n_beads/300 beads, at least 1.5) |
| `bw_files`<br>*ini only* | list of text | *(empty)* | BigWig tracks for extra binding energy (comma-separated) |
| `bw_strengths`<br>*ini only* | list of numbers | *(empty)* | Strength r per BigWig track (empty = -N_beads/10 each) |
| `lef_track_file`<br>`--lef-track` | text | *(empty)* | BigWig of LEF loading preference (cohesin/condensin ChIP) |
| `comp_bw_file`<br>*ini only* | text | *(empty)* | BigWig of compartments (A/B signal) |
| `comp_bed_file`<br>*ini only* | text | *(empty)* | BED of compartments / subcompartments |
| `min_pet`<br>`--min-pet` | integer (PET) | `0` | BEDPE: keep only loops with at least this many PET counts / score (0 = keep all) |
| `motif_genome`<br>`--motif-genome` | text | *(empty)* | BEDPE: reference genome FASTA; enables the CTCF motif search that sets anchor orientations |
| `motif_pfm`<br>`--motif-pfm` | text | *(empty)* | Motif PFM file (empty = bundled JASPAR CTCF MA0139.1) |
| `motif_pvalue`<br>`--motif-pvalue` | number (p-value) | `1e-05` | Motif hit cutoff: chance of a hit in random sequence (smaller = stricter) |
| `motif_pseudocount`<br>*ini only* | number | `0.8` | Pseudocount added to the PFM counts (avoids impossible bases) |
| `motif_pad`<br>*ini only* | integer (bp) | `0` | Extend every anchor by this many bp on both sides before the motif search |

### Output and reproducibility

Section `[output]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `out_path`<br>`--out` | text | *(empty)* | Results folder (empty = ./results in the current folder); each run gets its own subfolder |
| `run_number`<br>`--run` | integer | *(empty)* | Run number in the folder name (empty = next free; standalone stages reuse the latest) |
| `verbosity`<br>*ini only* | integer | `0` | Console detail: 0 main steps + progress bars, 1 + tables, 2 + info messages, 3 + debug (CLI: -v, -vv, -vvv) |
| `quiet`<br>`--quiet` | true / false | `False` | Print nothing to the console (the log file and the report are still written) |
| `seed`<br>`--seed` | integer | *(empty)* | Random seed (empty = a random one, saved in the resolved config); the same seed repeats the run exactly, MD included only with --threads 1 |
| `report`<br>`--report` | true / false | `True` | Write report.html |
| `figures`<br>`--figures` | true / false | `True` | Draw and save the figures (off: tables and data only, much faster for big or repeated runs) |
| `save_metadata`<br>`--metadata` | true / false | `True` | Save map matrices (MC and MD heatmaps, distances, strengths) as .npy in data/metadata |
| `save_ensemble`<br>`--save-ensemble` | true / false | `True` | Save the 3D structures: every MD frame as .cif, plus the start structure, topology and trajectory (structures/) |

### Parallelism and replicas

Section `[parallel]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `threads`<br>`--threads` | integer (threads) | `1` | Worker threads for the numba sampler, OpenMM CPU and numba kernels (0 = all cores) |
| `replicas`<br>`--n-replicas` | integer (chains) | `1` | Independent Monte Carlo chains (seed, seed+1, ...) run in parallel; the ensemble is pooled |

### Monte Carlo

Section `[monte_carlo]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `n_steps`<br>`--mc-steps` | integer (MC steps) | `20000` | Total Monte Carlo steps |
| `mc_step`<br>`--sample-every` | integer (MC steps) | `200` | Save one snapshot every this many steps |
| `burnin`<br>`--burnin` | integer (MC steps) | `1000` | Burn-in period discarded from statistics |
| `t_init`<br>`--t-init` | number (a.u.) | `3.0` | Initial temperature |
| `t_final`<br>`--t-final` | number (a.u.) | `1.0` | Final temperature (Annealing only) |
| `method`<br>`--method` | text | `Annealing` | Sampling method |
| `n_lef`<br>`--n-lef` | integer (LEFs) | *(empty)* | Loop extrusion factors (empty = N_anchors / 2) |
| `n_lef2`<br>*ini only* | integer (LEFs) | `0` | Second LEF family (different speed) |
| `lef_drift`<br>*ini only* | true / false | `False` | Unused; accepted only so older config files still load |
| `lef_rw`<br>`--lef-rw` | true / false | `True` | LEF legs slide as a random walk (True) or only extrude outward (False) |
| `lef_speed_dist`<br>`--lef-speed` | text | `constant` | Initial LEF speeds: constant (all equal), maxwell (Maxwell-Boltzmann), lognormal, or two_state (slow and fast LEFs); mean speed is always 1 |
| `lef_speed_spread`<br>`--lef-speed-spread` | number | `0.5` | lognormal only: width (sigma of ln speed) |
| `lef_fast_fraction`<br>`--lef-fast-fraction` | number | `0.2` | two_state only: fraction of fast LEFs |
| `lef_fast_ratio`<br>`--lef-fast-ratio` | number | `5.0` | two_state only: speed of fast LEFs relative to slow ones |
| `variable_sliding_step`<br>*ini only* | true / false | `False` | Sliding step uniform in [1, shortest loop] beads |
| `p_epigenetic`<br>*ini only* | number | `0.5` | Epigenetic model only: probability that a step changes a bead's epigenetic state instead of moving a LEF |

### Energy model coefficients

Section `[energy]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `folding_coeff`<br>*ini only* | number | `1.0` | Folding coefficient f |
| `folding_coeff2`<br>*ini only* | number | `0.0` | Folding coefficient f2 of the second family |
| `bind_coeff`<br>*ini only* | number | `1.0` | Anchor binding coefficient b |
| `cross_coeff`<br>*ini only* | number | `1.0` | LEF crossing coefficient kappa |
| `epi_coeff`<br>*ini only* | number | `0.0` | Epigenetic-model coupling strength; 0 = off (default: pure loop-extrusion model) |
| `n_epi_states`<br>*ini only* | integer | `3` | Number of epigenetic states |
| `epigenetic_mode`<br>*ini only* | text | `dynamic` | Epigenetic coupling J_ij: 'dynamic' = 1 on the backbone and on current LEF loops, else 0 (follows the LEFs); 'static' = fixed map from the loop data (advanced, config.ini only) |
| `cross_loop`<br>*ini only* | true / false | `True` | Penalise crossing loops, not only shared anchors |
| `between_families_penalty`<br>*ini only* | true / false | `True` | Apply crossing penalty between LEF families |
| `data_loss_mode`<br>*ini only* | integer | `0` | 0 = 1D anchor profiles L/R; 1 = 2D loop map |
| `loss_sigma`<br>*ini only* | number (beads) | `2.0` | Gaussian width of the 2D loop map J_loss when data_loss_mode = 1, so LEF legs feel a smooth landscape (advanced, config.ini only) |
| `binding_mode`<br>`--binding-mode` | text | `preferential` | Anchor binding mode: preferential = LEFs are attracted to the anchor profiles (non-negative), loop-free regions cost nothing; contrastive = in addition, loops are penalised for spanning loop-free regions and rewarded for spanning regions the input loops cover (needs loops) |

### Molecular dynamics

Section `[molecular_dynamics]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `device`<br>*ini only* | text | `` | Device index for CUDA or OpenCL (count from 0; empty = default) |
| `run_md`<br>`--run-MD` | true / false | `True` | Run the 3D molecular dynamics stage |
| `platform`<br>`--platform` | text | `auto` | OpenMM platform: auto (try CUDA, then OpenCL, then parallel CPU), CUDA, OpenCL, CPU or Reference |
| `initial_structure_type`<br>`--init-structure` | text | `rw` | Initial polymer shape |
| `integrator_step`<br>`--step-size` | number (fs) | `100.0` | Integrator time step |
| `sim_step`<br>`--md-steps` | integer (MD steps) | `5000` | MD steps per MC snapshot (per frame) |
| `sim_temp`<br>`--temperature` | number (K) | `310.0` | Simulation temperature |
| `friction`<br>*ini only* | number (1/ps) | `0.1` | Langevin friction |
| `tolerance`<br>*ini only* | number (kJ/mol/nm) | `0.001` | Energy-minimisation tolerance |
| `angle_ff_strength`<br>*ini only* | number (kJ/mol/rad^2) | `50.0` | Chain stiffness (angle force) |
| `le_ff_length`<br>*ini only* | number (nm) | `0.1` | Loop-anchor equilibrium distance |
| `le_ff_strength`<br>*ini only* | number (kJ/mol/nm^2) | `60000.0` | Loop-anchor spring constant |
| `ev_ff_strength`<br>*ini only* | number (kJ/mol) | `100.0` | Excluded-volume strength |
| `ev_ff_power`<br>*ini only* | number | `6.0` | Excluded-volume exponent (larger = harder, shorter-ranged wall; 6 keeps the chain from passing through itself without making it over-avoiding) |
| `do_compartments`<br>*ini only* | true / false | `False` | Add block-copolymer compartment force (needs epi_coeff > 0) |
| `comp_strength`<br>*ini only* | number | `0.25` | Block-copolymer compartment attraction: scale of the interaction table (1 = full strength of the table; lower = softer, compartments stay open instead of collapsing) |
| `comp_range`<br>*ini only* | number | `0.03` | Block-copolymer attraction range in units of sqrt(N) nm; smaller = shorter range |
| `md_skip_burnin`<br>*ini only* | true / false | `False` | Simulate only post-burn-in MC snapshots (False: every snapshot) |
| `forcefield_path`<br>*ini only* | text | `src/loopsage/molecular_dynamics/forcefields/classic_sm_ff.xml` | OpenMM force-field XML |

### 3D structure analysis and rendering

Section `[structure_3d]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `render_3d`<br>`--render-3d` | true / false | `True` | Write 3d_viewer.html and videos of the 3D trajectory |
| `make_videos`<br>*ini only* | true / false | `True` | Render mp4 videos (needs pyvista; ffmpeg via imageio-ffmpeg) |
| `colorings`<br>*ini only* | list of text | `purple, rainbow` | Video colourings: purple, rainbow, compartments (added automatically when compartments are loaded) |
| `representation`<br>*ini only* | text | `tube` | Video representation: tube (continuous chain) or beads (beads on a string); the viewer offers both |
| `interp_steps`<br>*ini only* | integer (frames) | `5` | Interpolated frames between two MD frames (smooth trajectories) |
| `video_fps`<br>*ini only* | integer (fps) | `30` | Video frame rate |
| `video_size`<br>*ini only* | integer (px) | `640` | Video width and height |
| `max_video_frames`<br>*ini only* | integer (frames) | `240` | Maximum frames per video |

### Analysis

Section `[analysis]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `corr_smooth`<br>*ini only* | number (beads) | `1.5` | Gaussian smoothing (beads) applied to per-position signals before the extra validation correlations (0 = off) |
| `track_smooth`<br>`--track-smooth` | number (beads) | `7.0` | Gaussian smoothing of the Hi-C insulation score and of the A/B track (Hi-C eigenvector or compartment BigWig) used as input and for validation (0 = off) |
| `contact_radius`<br>*ini only* | number (nm) | `1.0` | Contact distance for MD contact maps |
| `zoom_snapshots`<br>*ini only* | integer (snapshots) | `120` | Snapshots shown in the zoomed kymograph (window near equilibrium) |
| `zoom_start`<br>*ini only* | integer (snapshot) | *(empty)* | First snapshot of the zoom window (empty = middle of production) |

### Parameter optimizer

Section `[optimizer]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `optimize`<br>`--optimize` | true / false | `False` | Before the run, fit the number of LEFs, folding and binding coefficients with Nelder-Mead on short Monte Carlo runs (n_lef of the first family, f, b and the temperature schedule), then run with them |
| `opt_evals`<br>`--opt-evals` | integer (evaluations) | `60` | Maximum number of short Monte Carlo evaluations of the optimizer |
| `opt_early_stop`<br>`--opt-early-stop` | true / false | `True` | Stop the optimizer early when the loss no longer improves |
| `opt_interactive`<br>`--opt-interactive` | true / false | `True` | Pressing Enter during the optimizer shows the best parameters so far and asks whether to keep them and stop (only in a terminal) |
| `opt_patience`<br>`--opt-patience` | integer (evaluations) | `20` | Early stopping: evaluations without a real improvement (see opt_min_gain) before the optimizer stops |
| `opt_min_gain`<br>*ini only* | number | `0.01` | Early stopping: smallest drop of the best loss that counts as an improvement (the loss is about 1-4 at the start) |
| `opt_steps`<br>`--opt-steps` | integer (MC steps) | `4000` | Monte Carlo steps of one optimizer evaluation (each step is a sweep over all LEFs; the first quarter is burn-in, 200 snapshots; the final candidate check runs 3x longer) |
| `opt_max_beads`<br>`--opt-max-beads` | integer (beads) | `1500` | Largest region the optimizer simulates: bigger regions are optimized on the window of this many beads with most loops, and the number of LEFs is scaled back to the full region |
| `target_smooth`<br>`--target-smooth` | number (beads) | `0.8` | Smoothing of the optimizer target built from the input loops, so that the sparse loops look like a Hi-C map (0 = none) |
| `hic_denoise`<br>`--hic-denoise` | number (beads) | `1.0` | Denoising of the Hi-C target of the optimizer, to bring out the TAD pattern (0 = none); validation always uses the raw map |

### Topology

Section `[topology]` of the config file.

| Key / CLI option | Type (unit) | Default | What it does |
|---|---|---|---|
| `detect_knots`<br>`--knots` | true / false | `False` | Optional linking analysis of loops in 3D (slow; off by default) |
| `knots_stride`<br>*ini only* | integer (frames) | `10` | Use every n-th MD frame for linking numbers |
| `ev_p`<br>*ini only* | number | `0.0` | Topoisomerase ('random' mode): probability/fraction of beads with excluded volume off during a frame |
| `continuous_top`<br>*ini only* | true / false | `False` | Topoisomerase ('random' mode) acts on one continuous stretch |
| `topo_mode`<br>*ini only* | text | `random` | Topoisomerase model: 'random' (beads with EV off for a whole frame, set by ev_p) or 'local' (short strand-passage events at close contacts) |
| `topo_rate`<br>*ini only* | number (events) | `5.0` | 'local' mode: strand-passage attempts per MD frame per 1000 beads |
| `topo_radius`<br>*ini only* | number (bond lengths) | `2.0` | 'local' mode: strands closer than this can pass through each other |
| `topo_width`<br>*ini only* | integer (beads) | `3` | 'local' mode: beads of the passing strand that lose their excluded volume |
| `topo_window`<br>*ini only* | number (fraction) | `0.05` | 'local' mode: share of the frame's MD steps with excluded volume off, per attempt |
| `topo_keep`<br>*ini only* | number | `1.0` | 'local' mode: probability to keep a passage that creates a new link between loops (1 = no bias; below 1 the topoisomerase simplifies the topology) |
| `track_crossings`<br>*ini only* | true / false | `False` | Count chain crossings during the MD (checks whether the excluded volume leaks); also on with --knots |
| `track_every`<br>*ini only* | integer (MD steps) | `100` | MD steps between crossing checks |
| `ev_barrier_target`<br>*ini only* | number (kT) | `10.0` | Strand-crossing barrier used for the suggested ev_ff_strength in the topology report |
