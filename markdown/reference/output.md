# Output

Every run writes **one folder** that holds everything it produced: the report, the figures, the 3D structures, the saved data and the settings that were used. This page explains where each thing is, what the numbers in the files mean, how to switch outputs on and off, and how to reuse the saved data for your own analysis.

## The run folder

The output folder is **yours to choose**, with `--out` (`-o`, config key `out_path`). LoopSage creates it, with all its parent folders, if it does not exist yet, and writes the files into it exactly as given. The region is not part of the layout: the chromosome and the coordinates are written in the report and in `config_resolved.ini`, not in a folder name.

| You give | The results go to |
|---|---|
| `--out results` | `results/` |
| `--out results/chr1_134_489` | `results/chr1_134_489/` (`results/` is created too if needed) |
| `--out /data/hic/exp7/run_a` | that folder, created with all its parents |
| nothing | `results/<chrom>_<start>_<end>/` in the current folder, for example `results/chr1_77000000_95000000/` |

Without `--out` the region is added to the default folder, so that runs of different regions do not overwrite each other. An existing file with the name of the folder is an error.

The folder holds:

```text
results/                           the output folder, whatever you named it
├── report/
│   └── report.html                the full report: open this first
├── visualizations/                every figure, as .png and .svg
│   └── 3d/                        interactive viewer and videos
├── structures/                    3D structures of the polymer
│   └── ensemble/                  one .cif file per MD frame
├── data/                          saved results of each stage (for reuse)
│   └── metadata/                  small arrays: maps, optimized parameters
├── logs/
│   └── loopsage.log               everything the run printed, in detail
└── config_resolved.ini            every setting the run used
```

### Several replicas

With one replica (the default) the folders above are exactly what you get, with no `replica_<k>` subfolders. With `--n-replicas N` (N ≥ 2) every replica is a fully independent simulation: its own Monte Carlo chain (random start, own seed), and its own 3D run (own start structure, loops and random stream). The seeds follow from `--seed`: replica 0 uses it, the others get independent streams derived from it, and each is listed in the *Replicas* tables of the report. Everything stays in the same run folder; only the per-replica files get a `replica_<k>` subfolder (`k` = 0, 1, ...):

```text
results/
├── visualizations/
│   ├── replica_maps.png, md_replica_maps.png, ...   diagnostics across all replicas (also in the report)
│   └── replica_0/, replica_1/, ...                  figures of one replica, incl. its 3d/ viewer and videos
├── structures/
│   └── replica_0/, replica_1/, ...                  init_structure.cif, trajectory.dcd, ensemble/ of that replica
└── data/
    ├── mc_r0.npz, mc_r1.npz, ...  md_r0.npz, md_r1.npz, ...   cached chains and MD runs, one file per replica
    └── metadata/
        ├── mc_contact_map.npy, md_avg_distance.npy, ...   all replicas pooled
        ├── *_replica_std.npy                              spread between replicas
        └── replica_0/, replica_1/, ...                    the matrices of one replica
```

The report and the files in the main folders show the **pooled** result (all replicas together) plus the comparison between replicas; the per-replica figures are only saved as files, to keep the report short. The diagnostics across replicas are:

| Figure | What it shows |
|---|---|
| `replicas` | energy of each chain, loop occupancy and its correlation (convergence of the Monte Carlo) |
| `replica_maps` | mean loop contact map over replicas, its standard deviation between replicas, and the correlation of the replica maps |
| `md_replica_maps` | the same for the 3D contact maps |
| `md_replica_distances` | mean 3D distance map, its spread (std) between replicas and the relative spread std / mean |
| `md_replicas` | radius of gyration and potential energy along the MD frames of every replica |

**Standard deviation between replicas** is, for every pair of beads, the standard deviation of that value over the replicas' own averages: it is large where the result depends on the random start. The `*_replica_std.npy` files hold the same matrices.

### Repeated runs

Running again with the same `--out` writes into the same folder and **overwrites** its results (the log warns about it). To keep several runs, give each its own folder, for example `--out results/test1`, `--out results/test2`, or one per region, `--out results/chr1_134_489`.

!!! tip "Which run was which?"
    Name the folders as you like. What made the runs different is written in `config_resolved.ini` inside each folder (see below), so compare those files, not the names.

## What is in each folder

### `report/report.html`

A single, self-contained web page: open it in any browser, it works offline and can be sent by e-mail. It collects, in order, the input data, the optimizer (if it was used), the Monte Carlo loop extrusion, the 3D structures, the validation against Hi-C and the topology analysis. Every figure and table comes with a plain-language explanation of what it shows.

### `visualizations/`: the figures

All figures are saved twice with the same name: `name.png` (raster, for slides) and `name.svg` (vector, for papers). The files are flat in the folder, grouped below by the report section they belong to. The name is the *key* shown in the table.

| Section | Figure files (without extension) | What they show |
|---|---|---|
| Input data | `binding_profiles`, `anchor_map`, `anchor_stats`, `anchor_loops`, `loop_matrices`, `tracks`, `compartments`, `hic_input`, `motif_search` | the anchors and CTCF orientation, loop strength and length, the loop maps $J$ and $J^{\mathrm{loss}}$, extra tracks, compartments, the Hi-C map, the motif search |
| Optimizer | `optimizer`, `optimizer_target`, `optimizer_landscape` | the loss during the search, the target map, the parameter values the search visited |
| Monte Carlo | `energy_traces`, `convergence`, `lef_trajectories`, `kymograph`, `kymograph_zoom`, `occupancy_lengths`, `loop_landscape`, `mc_heatmap`, `loop_nesting`, `loop_parents_children`, `lef_velocity`, `lef_speed_init`, `lef_phase_space`, `lef_free_energy`, `lef_free_energy_3d` | energy and convergence, where the loop extruders (LEFs) go over time, loop sizes and occupancy, the simulated contact map, nesting, speeds, free-energy landscapes |
| Epigenetic model (only if `epi_coeff` is not 0) | `epigenetic_model`, `epigenetic_maps`, `epigenetic_features`, `epigenetic_loops`, `epigenetic_free_energy`, `epi_trajectory` | domains and energy, the checkerboard pattern, the effect on compartments, the free-energy landscape |
| 3D structures (MD) | `md_maps`, `md_series`, `md_geometry`, `mc_vs_md` | contact maps from 3D structures, compaction and energy during the run, geometry, loops versus 3D |
| 3D diagnostics | `struct_shape`, `struct_snapshots`, `struct_pairs`, `struct_scaling`, `struct_compartments` | shape over time, start, middle and end structures, pair distances, polymer scaling, compartments in 3D |
| Validation | `hic_validation`, `hic_correlation`, `hic_scores`, `validation_pearson`, `validation_pearson_direct`, `validation_ps_decay`, `validation_insulation`, `validation_insulation_anchor`, `validation_pc1`, `validation_summary` | agreement of the simulation with Hi-C or with the experimental loops: correlation, contact decay $P(s)$, insulation, compartments |
| Topology (only with `--knots`) | `links_map`, `links_time`, `links_dynamics`, `links_examples`, `links_approach`, `links_stills`, `crossings` | linking numbers between loops and chain crossings |

!!! note "Figures that do not appear"
    A figure is only drawn when the stage that makes it ran. No Hi-C file means no `hic_*` and fewer `validation_*` figures; `--no-run-MD` removes all 3D sections. This is normal.

### `visualizations/3d/`: viewer and videos

| File | What it is |
|---|---|
| `3d_viewer.html` | an interactive 3D player of the whole trajectory: rotate, zoom, play. One self-contained file with no downloads needed |
| `structure_<colouring>.mp4` | a video of the trajectory. One per colouring in `colorings`: `purple`, `rainbow`, and `compartments` (added automatically when compartment data exist) |
| `structure_<colouring>.png` | the poster image of that video |

### `structures/`: the 3D polymer

One bead of the model is one residue of the structure. The coordinates are in Ångström, as the CIF format expects (1 Å = 0.1 nm).

| File | What it is |
|---|---|
| `init_structure.cif` | the starting shape of the chain, before any simulation |
| `topology.psf` | the bond list, to load the trajectory in VMD or PyMOL |
| `trajectory.dcd` | the full MD trajectory, one frame every `sim_step` steps |
| `ensemble/MDLE_1.cif`, `MDLE_2.cif`, ... | one structure per MD frame, in time order. `MDLE` stands for *molecular dynamics loop extrusion*. Each file is built from one snapshot of the loop-extrusion simulation |

The `.cif` files open in ChimeraX, PyMOL and VMD.

### `logs/loopsage.log`

Everything the run reported, including the detail that the console hides at the default verbosity. The log always records the most detailed level, whatever `-v` you used.

### `config_resolved.ini`

Every setting of the run, in the same format as a config file, so you can **repeat the run exactly** with `loopsage -c config_resolved.ini`. Each setting carries a note on where its value came from: `default`, the config file, the command line, or `optimizer` (when the optimizer changed it). It also holds the **random seed** of the run, even if you did not choose one. This is the file to read when you compare runs.

## The saved data: `data/`

Each stage stores its result here, so that later steps and your own scripts can use it without repeating the computation. The files are compressed NumPy archives (`.npz`), plain JSON or tab-separated tables, so they open without LoopSage.

| File | Written by | Content |
|---|---|---|
| `inputs.npz` | input processing | the model inputs at bead resolution: binding profiles `L`, `R`; loop maps `J`, `J_loss`; `lengths`; `cover`; extra tracks `BWs`; `lef_track`; compartment signal `comp`; loop statistics in `meta` |
| `loops_annotated.bedpe` | motif finder | your loops with the CTCF orientation probabilities as columns 8 and 9 |
| `motif_loops.tsv`, `motif_anchors.tsv`, `motif_stats.json` | motif finder | one row per kept loop and per scanned anchor with the motif scores, and the counts at each filtering step |
| `optimization.json` | optimizer | the full record of the search |
| `mc.npz` | Monte Carlo | the loop-extrusion simulation (arrays below). With several replicas one file per replica instead: `mc_r0.npz`, `mc_r1.npz`, ... |
| `md.npz` | molecular dynamics | the 3D coordinates and maps (arrays below). With several replicas: `md_r0.npz`, `md_r1.npz`, ... |
| `links.npz` | topology | linking numbers per frame (only with `--knots`) |

Because these files exist, you can stop and continue a run (see below).

### Monte Carlo arrays (`mc.npz`)

A **snapshot** is the state of the simulation saved every `mc_step` Monte Carlo steps. There are $n_s$ snapshots, $N_{\mathrm{lef}}$ loop extruders and $N$ beads. Positions are **bead indices** (0 to $N-1$).

| Array | Shape | Meaning |
|---|---|---|
| `Ms` | $N_{\mathrm{lef}}\times n_s$ | left leg of every loop extruder over time |
| `Ns` | $N_{\mathrm{lef}}\times n_s$ | right leg. The loop of extruder $i$ at snapshot $t$ is the pair `(Ms[i,t], Ns[i,t])` |
| `Es` | $n_s$ | total energy of the snapshot (arbitrary units) |
| `Fs`, `Ks`, `Bs` | $n_s$ | the three energy terms: folding, crossing and binding (data) |
| `ufs` | $n_s$ | fraction of the chain that is not inside any loop (*unfolded fraction*) |
| `Ts` | $n_s$ | temperature at the snapshot (changes during Annealing) |
| `epi` | per bead and snapshot | epigenetic state of the beads over time (only meaningful with `epi_coeff` not 0) |
| `acc` | small | counts of proposed and accepted moves |
| `meta` | JSON text | the settings of this chain: `n_beads`, `n_lef`, `n_steps`, `mc_step`, `burnin`, `method`, `runtime_s`, `replicas`, `threads`, `seed`, ... |

The first `burnin` steps are the *burn-in*: the simulation is still forgetting its random start. Snapshots before index `burnin // mc_step` should not be averaged.

### Molecular dynamics arrays (`md.npz`)

There are $F$ frames (one per MC snapshot that was simulated).

| Array | Shape | Meaning |
|---|---|---|
| `coords` | $F\times N\times 3$ | bead positions in **nm** |
| `avg_map`, `std_map` | $N\times N$ | mean and standard deviation over frames of the contact strength: a smooth function of the distance $d$ that is 1 for touching beads and falls to 0 beyond the contact radius `contact_radius` (default Gaussian, $e^{-d^2/2\rho^2}$ with $\rho$ = `contact_radius`) |
| `avg_dist`, `std_dist` | $N\times N$ | mean and standard deviation over frames of the pairwise distance, in nm |
| `rg` | $F$ | radius of gyration of every frame, in nm (the size of the chain) |
| `epot` | $F$ | potential energy of every frame, in kJ/mol |
| `frame_idx` | $F$ | the Monte Carlo snapshot each frame was built from |
| `meta` | JSON text | `platform`, `runtime_s`, `n_frames`, `dt_fs` (time step), `steps_per_frame`, `ps_per_frame`, `temperature_K`, `contact_radius_nm`, ... |

## The metadata: `data/metadata/`

The *metadata* are the **small matrices and numbers that summarise a run**. They are the part you want for comparing many runs, so they are saved separately and are cheap to keep. All matrices are $N\times N$ on the bead grid of the region: row and column $i$ is the genomic window starting at `start + i × bp_per_bead`, where `bp_per_bead` = (end − start) / $N$.

| File | Content | Saved when |
|---|---|---|
| `mc_loop_counts.npy` | mean number of loops that join beads $i$ and $j$ over the post-burn-in ensemble (all replicas pooled) | `save_metadata` |
| `mc_contact_map.npy` | the loop map built from the same ensemble, with communication between loops (the map the optimizer scores) | `save_metadata` |
| `md_avg_distance.npy`, `md_std_distance.npy` | mean and standard deviation of the 3D distance in nm, over the MD frames | `save_metadata`, MD on |
| `md_avg_strength.npy`, `md_std_strength.npy` | mean and standard deviation of the contact strength over the MD frames | `save_metadata`, MD on |
| `mc_contact_map_replica_std.npy`, `md_avg_distance_replica_std.npy`, `md_avg_strength_replica_std.npy` | standard deviation between the replicas of the maps above | 2 or more replicas |
| `replica_<k>/` (same file names) | the matrices of replica `k` alone; the files above are then pooled over all replicas | 2 or more replicas |
| `optimized_parameters.json` | the values the optimizer chose (`chosen`: `n_lef`, `folding_coeff`, `bind_coeff`, temperatures), the starting point (`start`), the best result (`best`), whether it converged, number of evaluations and time | `--optimize`. Always saved, it is tiny |
| `hic_map.npy`, `hic_info.json` | the Hi-C reference map of the region and where it came from (file, normalisation, resolution) | a Hi-C file is given (`--hic`) |

## Switching outputs on and off

Each output has a switch. Boolean switches come in pairs on the command line (`--figures` and `--no-figures`), and every one is also a key in the `[output]` section of a config file. Keys without a flag are set in the config file, or on the command line with `--set KEY=VALUE`.

| What | Command line | Config key | Default | Effect when off |
|---|---|---|---|---|
| Report | `--no-report` | `report` | on | no `report/report.html` |
| Figures | `--no-figures` | `figures` | on | no `.png` / `.svg` files; tables and data are still produced. Much faster for big or repeated runs |
| Map matrices | `--no-metadata` | `save_metadata` | on | no `.npy` files in `data/metadata/` (the optimizer file stays) |
| 3D structures | `--no-save-ensemble` | `save_ensemble` | on | no `structures/` files (`.cif`, `.psf`, `.dcd`) |
| 3D stage altogether | `--no-run-MD` | `run_md` | on | no molecular dynamics, so no `structures/`, no `3d/`, no 3D diagnostics |
| Viewer and videos | `--no-render-3d` | `render_3d` | on | no `3d_viewer.html` and no videos |
| Videos only | `--set make_videos=false` | `make_videos` | on | the viewer is written, the `.mp4` files are not |
| Which videos | `--set colorings=purple` | `colorings` | `purple, rainbow` | only the listed colourings are rendered |
| Optimizer | `--optimize` | `optimize` | off | with it on: `optimization.json`, `optimized_parameters.json` and the optimizer figures |
| Topology | `--knots` | `detect_knots` | off | with it on: `links.npz` and the `links_*` figures |
| Console output | `-q`, `-v`, `-vv`, `-vvv` | `quiet`, `verbosity` | normal | `-q` prints nothing; the log file and report are still written |

Where the files go:

| What | Command line | Config key | Meaning |
|---|---|---|---|
| Output folder | `--out PATH`, `-o PATH` | `out_path` | the folder of the run, created with its parents and used as given (default `results/<chrom>_<start>_<end>`) |
| Random seed | `--seed N` | `seed` | repeat a run exactly. Without it a random seed is chosen and stored in `config_resolved.ini` |

A few examples:

```bash
# fastest useful run: data and tables only, no figures, no 3D
loopsage -c config.ini --no-figures --no-run-MD

# keep the 3D viewer but skip the videos and the large structure folder
loopsage -c config.ini --set make_videos=false --no-save-ensemble

# only the maps, a small result folder
loopsage -c config.ini --no-save-ensemble --no-render-3d --no-report
```

The same switches in a config file:

```ini
[output]
figures        = False
save_metadata  = True
save_ensemble  = False
report         = True
```

All keys, with their units and defaults, are in the [config reference](config.md); the command-line forms are in the [command-line reference](cli.md).

### Stopping and continuing

Every stage saves its result in `data/`. If you run a **single stage** again on an existing run, the stages before it are loaded from `data/` instead of being computed again, and only the chosen stage runs. For example, to redo the Monte Carlo stage with new settings on top of the saved inputs:

```bash
python -m loopsage.pipeline.stage_mc -c config.ini
```

The stage modules are `stage_preproc`, `stage_optimize`, `stage_mc`, `stage_md`, `stage_structure` and `stage_report` (all in `loopsage.pipeline`). The stages read the saved results from the folder given with `--out` (or the default folder of the region).

## Using the data in your own analysis

Everything above can be read without LoopSage. In Python:

```python
import json
import numpy as np

run = "results/chr1_77000000_95000000"                  # the folder you gave as --out

mc = np.load(f"{run}/data/mc.npz")
meta = json.loads(str(mc["meta"]))                   # run settings stored with the arrays
first = meta["burnin"] // meta["mc_step"]            # first snapshot after burn-in

lengths = (mc["Ns"] - mc["Ms"])[:, first:]           # loop lengths in beads, after burn-in
print(lengths.mean(), mc["Es"][first:].mean())       # mean loop length (beads) and mean energy

md = np.load(f"{run}/data/md.npz")
print(md["coords"].shape, md["rg"].mean())           # frames x beads x 3, mean radius of gyration (nm)

dist = np.load(f"{run}/data/metadata/md_avg_distance.npy")   # N x N mean 3D distance (nm)
```

LoopSage's own classes read the same files and handle the replicas for you: `MCResult.load(".../data/mc.npz")` (module `loopsage.monte_carlo.simulation`) and `MDResult.load(".../data/md.npz")` (module `loopsage.molecular_dynamics.engine`). The helper functions in `loopsage.results` (`loop_lengths`, `energies`, `contact_map`, `distance`, ...) then work on them; see the [Python interface](python.md).

### Comparing many runs (meta-analysis)

Because every run keeps its settings (`config_resolved.ini`) next to a small summary (`data/metadata/`), you can collect many runs into one table and see which settings change which result. This is also what [sensitivity scans](sensitivity.md) automate.

```python
import configparser, json
from pathlib import Path
import numpy as np
import pandas as pd

rows = []
for run in sorted(Path("results").glob("test*")):               # one --out folder per run, e.g. results/test1, results/test2
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cfg.read(run / "config_resolved.ini")
    mc = np.load(run / "data" / "mc.npz")
    meta = json.loads(str(mc["meta"]))
    first = meta["burnin"] // meta["mc_step"]
    rows.append(dict(
        run=run.name,
        n_lef=meta["n_lef"],
        folding=cfg["energy"]["folding_coeff"],      # a setting that differed between runs
        loop_beads=float((mc["Ns"] - mc["Ms"])[:, first:].mean()),
        energy=float(mc["Es"][first:].mean()),
        seed=cfg["output"]["seed"],
    ))

table = pd.DataFrame(rows)
print(table)                                              # one row per run, ready for plots or regression
```

Three habits make such a comparison reliable:

- **Compare after burn-in only**, as above, so the random start does not enter the numbers.
- **Use replicas.** With `--n-replicas 4` the four independent chains (and 3D runs) are stored in `mc_r0.npz` to `mc_r3.npz` (and `md_r0.npz` to `md_r3.npz`) and in `replica_<k>` folders; the spread between them is the noise level, and a difference smaller than that is not a result.
- **Fix the seed** (`--seed`) when you want a difference to come from a setting and not from chance. Monte Carlo is exactly repeatable with the same seed, and replica `k` is the same as a single run started with the seed listed for it; the 3D stage is repeatable only with `--threads 1`.
