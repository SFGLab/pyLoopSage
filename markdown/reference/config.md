# Config files

A config file keeps all the settings of a run in one place, so a run can be repeated exactly.
Pass it with `-c`:

```bash
python src/run.py -c config.ini
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
mc_steps = 40000
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

## All keys

*Type* tells how to write the value; *CLI option* gives the equivalent command-line option, if there is one.

## `[input]`: Input data and genomic region

| Key | Type (unit) | Default | CLI option | What it does |
|---|---|---|---|---|
| `interaction_file` | text | *(empty)* | `--file` | Loops/peaks file: .bedpe, .bed or .narrowPeak (required) |
| `chrom` | text | *(empty)* | *ini only* | Chromosome, e.g. chr1 (CLI: use --region) |
| `region_start` | integer (bp) | *(empty)* | *ini only* | Region start (empty = chromosome start) |
| `region_end` | integer (bp) | *(empty)* | *ini only* | Region end (empty = chromosome end) |
| `genome` | text | `hg38` | `--genome` | Genome assembly for chromosome sizes |
| `n_beads` | integer (beads) | *(empty)* | `--n-beads` | Number of simulation beads (empty = region / 2 kb) |
| `smoothing_input` | true / false | `False` | *ini only* | Smooth the anchor/peak signal with a Gaussian |
| `bw_files` | list of text | *(empty)* | *ini only* | BigWig tracks for extra binding energy (comma-separated) |
| `bw_strengths` | list of numbers | *(empty)* | *ini only* | Strength r per BigWig track (empty = -N_beads/10 each) |
| `lef_track_file` | text | *(empty)* | *ini only* | BigWig of LEF loading preference (cohesin/condensin ChIP) |
| `comp_bw_file` | text | *(empty)* | *ini only* | BigWig of compartments (A/B signal) |
| `comp_bed_file` | text | *(empty)* | *ini only* | BED of compartments / subcompartments |
| `min_pet` | integer (PET) | `0` | `--min-pet` | BEDPE: keep only loops with at least this many PET counts / score (0 = keep all) |
| `motif_genome` | text | *(empty)* | `--motif-genome` | BEDPE: reference genome FASTA; enables the CTCF motif search that sets anchor orientations |
| `motif_pfm` | text | *(empty)* | `--motif-pfm` | Motif PFM file (empty = bundled JASPAR CTCF MA0139.1) |
| `motif_pvalue` | number (p-value) | `1e-05` | `--motif-pvalue` | Motif hit cutoff: chance of a hit in random sequence (smaller = stricter) |
| `motif_pseudocount` | number | `0.8` | *ini only* | Pseudocount added to the PFM counts (avoids impossible bases) |
| `motif_pad` | integer (bp) | `0` | *ini only* | Extend every anchor by this many bp on both sides before the motif search |

## `[output]`: Output and reproducibility

| Key | Type (unit) | Default | CLI option | What it does |
|---|---|---|---|---|
| `out_path` | text | *(empty)* | `--out` | Results folder (empty = <project>/results); each run gets its own subfolder |
| `run_number` | integer | *(empty)* | `--run` | Run number in the folder name (empty = next free; standalone stages reuse the latest) |
| `verbosity` | integer | `0` | *ini only* | Console detail: 0 main steps + progress bars, 1 + tables, 2 + info messages, 3 + debug (CLI: -v, -vv, -vvv) |
| `seed` | integer | *(empty)* | `--seed` | Random seed (empty = random) |
| `report` | true / false | `True` | `--report` | Write report.html |
| `save_metadata` | true / false | `True` | `--metadata` | Save map matrices (MC and MD heatmaps, distances, strengths) as .npy in data/metadata |
| `save_ensemble` | true / false | `True` | `--save-ensemble` | Save every MD frame as .cif |

## `[parallel]`: Parallelism and replicas

| Key | Type (unit) | Default | CLI option | What it does |
|---|---|---|---|---|
| `threads` | integer (threads) | `1` | `--threads` | Worker threads for the numba sampler, OpenMM CPU and numba kernels (0 = all cores) |
| `replicas` | integer (chains) | `1` | `--n-replicas` | Independent Monte Carlo chains (seed, seed+1, ...) run in parallel; the ensemble is pooled |

## `[monte_carlo]`: Monte Carlo (loop extrusion) sampler

| Key | Type (unit) | Default | CLI option | What it does |
|---|---|---|---|---|
| `n_steps` | integer (MC steps) | `40000` | `--mc-steps` | Total Monte Carlo steps |
| `mc_step` | integer (MC steps) | `200` | `--sample-every` | Save one snapshot every this many steps |
| `burnin` | integer (MC steps) | `1000` | `--burnin` | Burn-in period discarded from statistics |
| `t_init` | number (a.u.) | `2.0` | `--t-init` | Initial temperature |
| `t_final` | number (a.u.) | `1.0` | `--t-final` | Final temperature (Annealing only) |
| `method` | text | `Annealing` | `--method` | Sampling method |
| `n_lef` | integer (LEFs) | *(empty)* | `--n-lef` | Loop extrusion factors (empty = N_anchors / 2) |
| `n_lef2` | integer (LEFs) | `0` | *ini only* | Second LEF family (different speed) |
| `lef_drift` | true / false | `False` | *ini only* | Kept for compatibility with the original LoopSage (has no effect there either) |
| `lef_rw` | true / false | `True` | `--lef-rw` | LEF legs slide as a random walk (True, original) or only extrude outward (False) |
| `lef_speed_dist` | text | `constant` | `--lef-speed` | Initial LEF speeds: constant (all equal, original), maxwell (Maxwell-Boltzmann), lognormal, or two_state (slow and fast LEFs); mean speed is always 1 |
| `lef_speed_spread` | number | `0.5` | `--lef-speed-spread` | lognormal only: width (sigma of ln speed) |
| `lef_fast_fraction` | number | `0.2` | `--lef-fast-fraction` | two_state only: fraction of fast LEFs |
| `lef_fast_ratio` | number | `5.0` | `--lef-fast-ratio` | two_state only: speed of fast LEFs relative to slow ones |
| `variable_sliding_step` | true / false | `False` | *ini only* | Sliding step uniform in [1, shortest loop] beads |
| `p_epigenetic` | number | `0.5` | *ini only* | Epigenetic model only: probability that a step changes a bead's epigenetic state instead of moving a LEF |

## `[energy]`: Energy model coefficients

| Key | Type (unit) | Default | CLI option | What it does |
|---|---|---|---|---|
| `folding_coeff` | number | `1.0` | *ini only* | Folding coefficient f |
| `folding_coeff2` | number | `0.0` | *ini only* | Folding coefficient f2 of the second family |
| `bind_coeff` | number | `1.0` | *ini only* | Anchor binding coefficient b |
| `cross_coeff` | number | `1.0` | *ini only* | LEF crossing coefficient kappa |
| `epi_coeff` | number | `0.0` | *ini only* | Epigenetic-model coupling strength; 0 = off (default: pure loop-extrusion model) |
| `n_epi_states` | integer | `3` | *ini only* | Number of epigenetic states |
| `epigenetic_mode` | text | `dynamic` | *ini only* | Epigenetic coupling J_ij: 'dynamic' = 1 on the backbone and on current LEF loops, else 0 (follows the LEFs); 'static' = fixed map from the loop data (advanced, config.ini only) |
| `cross_loop` | true / false | `True` | *ini only* | Penalise crossing loops, not only shared anchors |
| `between_families_penalty` | true / false | `True` | *ini only* | Apply crossing penalty between LEF families |
| `data_loss_mode` | integer | `0` | *ini only* | 0 = 1D anchor profiles L/R; 1 = 2D loop map |
| `contrastive_binding` | true / false | `False` | *ini only* | Penalise loop-free regions (tanh-contrast signals) |

## `[molecular_dynamics]`: Molecular dynamics (3D structures)

| Key | Type (unit) | Default | CLI option | What it does |
|---|---|---|---|---|
| `device` | text | `` | *ini only* | Device index for CUDA or OpenCL (count from 0; empty = default) |
| `run_md` | true / false | `True` | `--run-MD` | Run the 3D molecular dynamics stage |
| `platform` | text | `auto` | `--platform` | OpenMM platform: auto (try CUDA, then OpenCL, then parallel CPU), CUDA, OpenCL, CPU or Reference |
| `initial_structure_type` | text | `rw` | `--init-structure` | Initial polymer shape |
| `integrator_step` | number (fs) | `100.0` | `--step-size` | Integrator time step |
| `sim_step` | integer (MD steps) | `2000` | `--md-steps` | MD steps per MC snapshot (per frame) |
| `sim_temp` | number (K) | `310.0` | `--temperature` | Simulation temperature |
| `friction` | number (1/ps) | `0.1` | *ini only* | Langevin friction |
| `tolerance` | number (kJ/mol/nm) | `0.001` | *ini only* | Energy-minimisation tolerance |
| `angle_ff_strength` | number (kJ/mol/rad^2) | `100.0` | *ini only* | Chain stiffness (angle force) |
| `le_ff_length` | number (nm) | `0.1` | *ini only* | Loop-anchor equilibrium distance |
| `le_ff_strength` | number (kJ/mol/nm^2) | `60000.0` | *ini only* | Loop-anchor spring constant |
| `ev_ff_strength` | number (kJ/mol) | `100.0` | *ini only* | Excluded-volume strength |
| `ev_ff_power` | number | `3.0` | *ini only* | Excluded-volume exponent |
| `ev_p` | number | `0.0` | *ini only* | Topoisomerase: probability/fraction with excluded volume off |
| `continuous_top` | true / false | `False` | *ini only* | Topoisomerase acts on one continuous stretch |
| `do_compartments` | true / false | `False` | *ini only* | Add block-copolymer compartment force (needs epi_coeff > 0) |
| `md_skip_burnin` | true / false | `False` | *ini only* | Simulate only post-burn-in MC snapshots (original LoopSage uses every snapshot) |
| `forcefield_path` | text | `/mnt/attach/outputs/pyLoopSage/src/molecular_dynamics/forcefields/classic_sm_ff.xml` | *ini only* | OpenMM force-field XML |

## `[structure_3d]`: 3D structure analysis and rendering

| Key | Type (unit) | Default | CLI option | What it does |
|---|---|---|---|---|
| `render_3d` | true / false | `True` | `--render-3d` | Write 3d_viewer.html and videos of the 3D trajectory |
| `make_videos` | true / false | `True` | *ini only* | Render mp4 videos (needs pyvista; ffmpeg via imageio-ffmpeg) |
| `colorings` | list of text | `purple, rainbow` | *ini only* | Video colourings: purple, rainbow, compartments (added automatically when compartments are loaded) |
| `representation` | text | `tube` | *ini only* | Video representation: tube (continuous chain) or beads (beads on a string); the viewer offers both |
| `interp_steps` | integer (frames) | `5` | *ini only* | Interpolated frames between two MD frames (smooth trajectories) |
| `video_fps` | integer (fps) | `30` | *ini only* | Video frame rate |
| `video_size` | integer (px) | `640` | *ini only* | Video width and height |
| `max_video_frames` | integer (frames) | `240` | *ini only* | Maximum frames per video |

## `[analysis]`: Analysis

| Key | Type (unit) | Default | CLI option | What it does |
|---|---|---|---|---|
| `corr_smooth` | number (beads) | `1.5` | *ini only* | Gaussian smoothing (beads) applied to per-position signals before the extra validation correlations (0 = off) |
| `contact_radius` | number (nm) | `1.0` | *ini only* | Contact distance for MD contact maps |
| `detect_knots` | true / false | `False` | `--knots` | Optional linking analysis of loops in 3D (slow; off by default) |
| `zoom_snapshots` | integer (snapshots) | `120` | *ini only* | Snapshots shown in the zoomed kymograph (window near equilibrium) |
| `zoom_start` | integer (snapshot) | *(empty)* | *ini only* | First snapshot of the zoom window (empty = middle of production) |
| `knots_stride` | integer (frames) | `10` | *ini only* | Use every n-th MD frame for linking numbers |
