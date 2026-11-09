# Python interface

After `pip install pyLoopSage`, LoopSage can be used from any Python script, notebook or console. The package on PyPI is called **pyLoopSage**; the name to import is **`loopsage`**.

```bash
pip install pyLoopSage           # add --pre while the release is a release candidate
python -c "import loopsage as ls; print(ls.__version__)"
```

The Python interface is a set of **independent functions, one per step**: read the inputs, optimize, run the Monte Carlo, run the molecular dynamics, extract results, draw figures. You call them like any other Python functions and get Python objects back (arrays, tables, figures). **Nothing is written to disk** unless you pass a folder or call `ls.run`. The `loopsage` command and the config file work independently of this interface.

| Step | Function | Returns |
|---|---|---|
| settings | `ls.config(...)` | a `Config` |
| grid | `ls.geometry(cfg)` | region and bead grid |
| inputs | `ls.read_inputs(cfg, geom, loops=...)` | `InputData` |
| Hi-C target (optional) | `ls.read_hic(cfg, geom, path)` | `(matrix, info)` |
| optimizer (optional) | `ls.optimize(cfg, inputs, geom)` and `ls.apply_optimum(cfg, result)` | `OptResult`, a new `Config` |
| loop extrusion | `ls.run_mc(cfg, inputs, geom)` | `MCResult` |
| 3D structures | `ls.run_md(cfg, mc, geom)` | `MDResult` |
| results | `ls.results.<function>(mc, geom)` | arrays and tables |
| figures | `ls.plots.<function>(...)` | matplotlib figures |
| the whole command | `ls.run(...)`, `ls.main([...])` | the run folder is written |

## A first example

```python
import loopsage as ls

cfg = ls.config(region="chr1:77000000-95000000", n_beads=120, n_steps=20000, seed=1)   # all the settings in one Config object
geom = ls.geometry(cfg)                                          # the bead grid: 120 beads of 150 kb
inputs = ls.read_inputs(cfg, geom, loops="loops.bedpe")          # a file path: the loops (BEDPE)

mc = ls.run_mc(cfg, inputs, geom)                                # Monte Carlo loop extrusion

ls.results.mean_loop_length(mc, geom)                            # average loop length (kb) at every snapshot
ls.results.trajectories(mc, geom, "mb")                          # positions of the cohesin legs over time
ls.plots.energy_traces(mc)                                       # a matplotlib figure
```

Each line is a separate call, so a notebook can run them one at a time, repeat one with other settings, or stop after any of them.

## Settings

All settings of a run live in one object, a **`Config`**. In Python there is no config file to edit: you create the object with `ls.config(...)` and pass it to the functions. The same object is used by every step, so the inputs, the optimizer, the Monte Carlo and the molecular dynamics always see the same settings.

### Create it

Every parameter is a keyword argument, by its key name (`n_steps`, `folding_coeff`, `run_md`). Anything you leave out has its default value.

```python
cfg = ls.config(
    region="chr1:77000000-95000000",     # the genomic region in bp (shortcut for chrom, region_start, region_end)
    n_beads=120,                         # resolution: the region is cut into 120 beads
    n_steps=20000,                       # Monte Carlo steps
    n_lef=8,                             # number of loop extruders
    folding_coeff=1.5,                   # strength of the folding energy
    seed=1,                              # random seed: the same seed gives the same result
)
```

A command-line option can have a different name than the key: `--mc-steps` sets `n_steps` and `--md-steps` sets `sim_step`. The tables of the [command line](cli.md) and the [config file](config.md) show both names side by side, with the default of every parameter.

### Read it

Every parameter is an attribute. `cfg.source` says where each value came from: `default` for the values you left out, `cli` for the ones you gave as arguments, and `cfg.as_dict()` returns everything as a dictionary.

```python
cfg.n_steps                  # 20000
cfg.n_lef                    # 8
cfg.burnin                   # a default value
cfg.source["n_steps"]        # where n_steps came from
cfg.as_dict()                # all parameters as a dictionary
```

### Change it

To change settings, make a copy with new values. `ls.config(cfg, ...)` checks the new values and leaves `cfg` untouched, which makes it easy to compare settings:

```python
cfg_long = ls.config(cfg, n_steps=100000, seed=2)      # a validated copy with two changes
cfg_nomd = ls.config(cfg, run_md=False)

for f in (0.5, 1.0, 2.0):                              # a parameter scan
    mc = ls.run_mc(ls.config(cfg, folding_coeff=f), inputs, geom)
```

You can also assign an attribute directly (`cfg.n_steps = 5000`), but this is not checked: a misspelled name silently creates a new attribute that is ignored, and an invalid value is found only later. Use `ls.config(cfg, ...)` when in doubt.

### Find a parameter

```python
ls.parameters("md steps")        # search by name or description: {name: (default, help)}
ls.parameters()["n_steps"][0]    # the default of one parameter: 20000
```

### Values and errors

Values are converted as in a config file: `run_md="yes"`, `integrator_step="100 fs"` and `bw_files=["a.bw", "b.bw"]` all work. A wrong name or an invalid value (for example a burn-in larger than the number of steps) raises `ValueError` with a clear message, so a script can catch it.

### From a config file or a dictionary

A config file is optional. If you already have one, `ls.config` reads it, and your keyword arguments win over its values. A dictionary works the same way.

```python
cfg = ls.config("config.ini", n_steps=5000)                    # the file, with one key overridden
cfg = ls.config({"region": "chr1:1-9000000", "n_beads": 100}, seed=3)     # a dictionary
ls.write_ini("used.ini", cfg.as_dict(), cfg.source)            # save the settings as a config file (optional)
```

### The grid

`ls.geometry(cfg)` turns the region and `n_beads` into the bead grid used by all steps. It has `chrom`, `start`, `end`, `n_beads`, `bp_per_bead` and `bead_of(bp)`, which gives the bead that holds a genomic position. Without `n_beads`, one bead is about 2 kb.

## Inputs

Inputs are given as **strings with the path of the file**, the same files the command uses:

```python
inputs = ls.read_inputs(
    cfg, geom,
    loops="loops.bedpe",                        # loops: BEDPE with anchors, score and optional CTCF orientation
    compartments="compartments.bw",             # A/B signal: .bw or .bed
    tracks=["h3k27ac.bw", "rad21.bw"],          # extra binding tracks: one path or a list
    lef_track="loading.bw",                     # where LEFs prefer to load
)
```

Use `sites="peaks.narrowPeak"` instead of `loops` for binding sites without loop pairs. Without `loops` or `sites`, the config key `interaction_file` is used, so `ls.config("config.ini")` followed by `ls.read_inputs(cfg, geom)` reads the inputs named in the config file.

The result `InputData` has `loops` (the profiles `L`, `R`, the loop matrix `J`, `cover`), `BWs` (tracks, one row per track), `lef_track`, `comp` (compartments in [-1, 1]) and `comp_source`.

**Hi-C reference.** `ls.read_hic(cfg, geom, "map.hic")` returns `(matrix, info)`: the Hi-C map on the bead grid. It is the optimizer target.

**Inputs as arrays.** For synthetic data, give arrays instead of paths: `loops=(left, right, weight)` with positions in bp, `sites=dict(position=..., weight=..., reverse=...)`, and per-bead arrays (one value per bead) for `tracks`, `lef_track` and `compartments`. Compartment values must lie between -1 and 1 (A positive, B negative); other values raise `ValueError`. The [sandbox](#sandbox) builds such arrays conveniently.

## Optimizer

The optimizer searches the number of LEFs `n_lef`, the folding coefficient, the binding coefficient and the temperature by Nelder&ndash;Mead, so the simulated map matches a target: the Hi-C map when given, otherwise the loops in `inputs`.

```python
hic = ls.read_hic(cfg, geom, "map.hic")                         # optional target
result = ls.optimize(cfg, inputs, geom, hic=hic)                # a search of cfg.opt_evals evaluations
result.best                                                     # {'K': ..., 'f': ..., 'b': ..., 'T': ...}
cfg_opt = ls.apply_optimum(cfg, result)                         # a copy of cfg with the optimized values
mc = ls.run_mc(cfg_opt, inputs, geom)
ls.plots.convergence_figure(result)                             # how the search converged
```

## Monte Carlo

```python
mc = ls.run_mc(cfg, inputs, geom)
mc.Ms, mc.Ns        # left and right leg (bead number) of every LEF, arrays (LEFs, snapshots)
mc.Es               # energy at every snapshot
mc.chains           # all replicas (cfg.replicas), each with the same fields
```

A snapshot is saved every `mc_step` steps; the first `burnin` steps are the equilibration. The same `seed` gives the same result.

## Results

Functions that turn an `MCResult` into the quantities one usually needs. `geom` gives the genomic coordinates. Series cover all snapshots; `production=True` keeps the snapshots after the burn-in. `units` is `bead`, `bp`, `kb` or `mb`.

```python
r = ls.results
left, right = r.trajectories(mc, geom, "mb")        # cohesin legs over time, arrays (LEFs, snapshots)
r.loop_lengths(mc, geom, "kb")                      # loop length of every LEF, (LEFs, snapshots)
r.mean_loop_length(mc, geom, "kb", production=True) # average loop length per snapshot
r.energies(mc)                                      # table: step, total, folding, binding, crossing, temperature, unfolded_fraction
r.loops_table(mc, geom)                             # the loops of the last snapshot as a table (anchors in beads and bp)
r.loop_coverage(mc, geom)                           # average number of loops spanning each bead
r.leg_density(mc, geom)                             # average number of LEF legs on each bead
r.contact_map(mc, geom)                             # contact map implied by the loops, (beads, beads)
r.steps(mc)                                         # Monte Carlo step of every snapshot
```

## 3D structures

```python
md = ls.run_md(cfg, mc, geom)                       # OpenMM molecular dynamics from the Monte Carlo loops
md.coords                                           # positions in nm, (frames, beads, 3)

r.coordinates(md, frame=0)                          # one frame, (beads, 3)
r.radius_of_gyration(md)                            # radius of gyration (nm) of every frame
r.distance(md, geom, 80_000_000, 85_000_000, units="bp")   # distance (nm) between two positions in every frame
r.contact_map_md(md)                                # average contact map of the structures, (beads, beads)
```

The platform, steps and force-field settings are config keys (`platform`, `sim_step`, `tolerance`, ...). Without `folder`, `run_md` writes nothing; `ls.run_md(cfg, mc, geom, folder="structures_out")` also saves the structures and maps there.

## Figures

Each figure function of LoopSage is available by name and **returns a matplotlib figure** (white background) to show or save:

```python
fig = ls.plots.energy_traces(mc)                    # energy and its components over time
fig.savefig("energy.png", dpi=200)

ls.plots.kymograph(mc, geom, inputs)                # LEF positions over time
ls.plots.loop_landscape(mc, geom, inputs)           # loops as arcs
ls.plots.lef_trajectories(mc, geom)                 # tracks of single LEFs
ls.plots.binding_profiles(inputs, geom)             # the binding profiles L and R
ls.plots.loop_matrices(inputs, geom)                # the loop matrices of the inputs
ls.plots.convergence_figure(result)                 # optimizer convergence
```

All figure functions of the modules in `loopsage.viz` (`mc_figs`, `map_figs`, `input_figs`, `optimizer_figs`, `structure_figs`, ...) can be called this way.

## Saving

Functions write nothing by default. To keep results, save what you want: `np.save("map.npy", r.contact_map(mc, geom))`, `r.energies(mc).to_csv("energies.csv")`, `fig.savefig(...)`, or `mc.save("mc.npz")` and `md.save("md.npz")` for the full result objects (load them again with `MCResult.load` from `loopsage.monte_carlo.simulation` and `MDResult.load` from `loopsage.molecular_dynamics.engine`).

To write a complete run folder (report, figures, structures, data, exactly like the command), use `ls.run`:

```python
ctx = ls.run("config.ini", n_steps=20000)            # the pipeline of the command; ctx.dirs.root is the run folder
ctx = ls.run(interaction_file="loops.bedpe", region="chr1:77000000-95000000", out_path="results")
```

`ls.run` takes the same arguments as `ls.config` and returns the run context (`ctx.mc`, `ctx.md`, `ctx.inputs`, ...). Like the command, it saves every output unless it is switched off, for example with `report=False` or `figures=False` (the [output switches](cli.md) list them all).

## The command from Python

```python
ls.main(["-c", "config.ini", "-vv"])        # exactly `loopsage -c config.ini -vv`
```

`python -m loopsage -c config.ini` does the same from a shell. `main` takes the list of command-line words; mistakes in the options print the usual usage message and exit.

## Sensitivity scans

`ls.sweep` runs a parameter scan on a synthetic polymer and writes the combined `sensitivity_report.html`, the same as the `loopsage-scan` command. It returns a result whose `.table` holds one row per run. See the [sensitivity page](sensitivity.md).

```python
res = ls.sweep({"T": "lin(1, 16, 12)", "n_lef": [4, 12]}, system={"n_beads": 300}, out="my_scan")
```

## Importing modules

Every part of LoopSage is importable under the name `loopsage`, for example `import loopsage.monte_carlo as mc` or `from loopsage.preproc.loader import load_inputs`:

| Package | Contents |
|---|---|
| `loopsage.settings` | the parameters, config files, `make_config` and `load_config` |
| `loopsage.preproc` | reading loops, peaks, tracks and `.hic` files; `arrays` builds inputs from variables |
| `loopsage.motif_finder` | CTCF motif search and orientation |
| `loopsage.monte_carlo` | energies, moves and the sampler (`simulation.run_mc`) |
| `loopsage.optimizer` | loss functions and the Nelder&ndash;Mead search |
| `loopsage.molecular_dynamics` | OpenMM 3D simulation |
| `loopsage.analysis`, `loopsage.validation` | maps, features and the comparison with experiment |
| `loopsage.knots` | linking numbers |
| `loopsage.viz`, `loopsage.viz3d`, `loopsage.reporting` | figures, the viewer and the report |
| `loopsage.pipeline` | the stages and the runner |

The functions above are thin wrappers over these modules, which can be used directly when more control is needed:

```python
from loopsage.settings.loader import make_config
from loopsage.preproc.loader import build_geometry, load_inputs
from loopsage.monte_carlo.simulation import run_mc
from loopsage.analysis.maps import loop_map

cfg = make_config(region="chr1:77000000-95000000", n_beads=120, n_steps=20000, seed=1)
geom = build_geometry(cfg)
inputs = load_inputs(cfg, geom, "loops.bedpe")
mc = run_mc(cfg, inputs, geom, quiet=True)
cmap = loop_map(mc, geom.n_beads)
```

`import loopsage` is fast: the heavy modules (Numba, OpenMM, plotting) load when you first use them. The names are the same objects as in the source tree, so `loopsage.monte_carlo` and `monte_carlo` are one module.

## Function reference

### config

```python
ls.config(config=None, region=None, **parameters) -> Config
```

| Input | Meaning |
|---|---|
| `config` | `.ini` path, a `Config` or a dictionary (optional) |
| `region` | `"chr1:77000000-95000000"` (bp) |
| `**parameters` | any config key, e.g. `n_beads`, `n_steps`, `seed`; they override `config` |

Output: a validated `Config`. Raises `ValueError` for unknown names or invalid values.

```python
cfg = ls.config(region="chr1:1-9000000", n_beads=100, run_md=False)
```

### read_inputs

```python
ls.read_inputs(cfg, geom=None, loops=None, sites=None, compartments=None,
               tracks=None, track_names=None, lef_track=None) -> InputData
```

| Input | Meaning |
|---|---|
| `loops` | `.bedpe` path, or arrays `(left, right, weight)` / `dict(left, right, weight, left_reverse, right_reverse)` in bp |
| `sites` | `.bed` / `.narrowPeak` path, or `dict(position, weight, reverse)`; `reverse` is the probability that the CTCF points right (0 forward, 1 reverse, -1 unknown) |
| `compartments` | `.bw` / `.bed` path, or one value per bead in [-1, 1] |
| `tracks`, `track_names` | a `.bw` path or list of paths, or per-bead arrays |
| `lef_track` | `.bw` path, or one non-negative value per bead |

Output: `InputData` (`loops.L`, `loops.R`, `loops.J`, `BWs`, `lef_track`, `comp`).

```python
inputs = ls.read_inputs(cfg, geom, loops="loops.bedpe", compartments="comp.bw")
```

### optimize and apply_optimum

```python
ls.optimize(cfg, inputs, geom=None, hic=None) -> OptResult
ls.apply_optimum(cfg, result) -> Config
```

`hic` is the target from `ls.read_hic`; without it the loops of `inputs` (from a BEDPE file) are the target. Settings: `opt_evals`, `opt_steps`, `opt_patience`. The result has `best`, `start` and `potts` (when the epigenetic model is fitted). `apply_optimum` returns a copy of `cfg` with `n_lef`, `folding_coeff`, `bind_coeff`, `t_init`, `t_final` (and `epi_coeff`, `p_epigenetic`) set.

### run_mc

```python
ls.run_mc(cfg, inputs, geom=None) -> MCResult
```

Fields of the result (arrays have one column per snapshot):

| Field | Content |
|---|---|
| `Ms`, `Ns` | left and right leg of every LEF, `(LEFs, snapshots)` |
| `Es`, `Fs`, `Bs`, `Ks` | total, folding, binding and crossing energy |
| `Ts`, `ufs` | temperature and fraction of unfolded beads |
| `chains` | all replicas (`replicas` in the config) |
| `production(a)` | the snapshots of `a` after the burn-in |
| `pooled()` | all replicas joined |

### run_md

```python
ls.run_md(cfg, mc, geom=None, folder=None) -> MDResult
```

Runs the 3D simulation on the pooled Monte Carlo loops. Fields: `coords` (frames, beads, 3 in nm), `avg_map` (average contact map), `rg` (radius of gyration), `epot` (potential energy). With `folder`, structures and maps are saved there.

### results

```python
ls.results.<function>(mc, geom, ...)
```

| Function | Output |
|---|---|
| `trajectories(mc, geom, units="bead", production=False, chain=0)` | `(left, right)` arrays `(LEFs, snapshots)` |
| `loop_lengths(mc, geom, units="kb", ...)` | array `(LEFs, snapshots)` |
| `mean_loop_length(mc, geom, units="kb", ...)` | array `(snapshots,)` |
| `energies(mc, production=False, chain=0)` | `DataFrame` |
| `loops_table(mc, geom, snapshot=-1)` | `DataFrame` of one snapshot |
| `loop_coverage(mc, geom)`, `leg_density(mc, geom)` | arrays `(beads,)` |
| `contact_map(mc, geom)` | array `(beads, beads)` |
| `coordinates(md, frame=None)`, `radius_of_gyration(md)`, `distance(md, geom, a, b, units="bead")`, `contact_map_md(md)` | 3D results |

### plots

```python
ls.plots.<name>(...) -> Figure
```

Calls the figure function `<name>` of `loopsage.viz` and returns the matplotlib figure (a list when the function draws several).

## Sandbox

The sandbox builds a small artificial system by hand and feeds the same functions. It is useful to test an idea: what does a single CTCF pair do, what do compartments change?

```python
import numpy as np
import loopsage as ls

box = ls.Sandbox("chr1:0-3000000", n_beads=100)          # 3 Mb, 100 beads of 30 kb

# CTCF motifs: position (bp), orientation and strength
box.add_ctcf(300_000, "forward", strength=10)            # forward motif: left anchor of a loop
box.add_ctcf(900_000, "reverse", strength=8)             # reverse motif: right anchor
box.add_ctcfs([1_500_000, 2_400_000], ["forward", "reverse"], [6, 9])      # several at once
box.add_ctcf(2_000_000, 0.3)                             # or a probability of pointing right

# compartments: your own array, one value per bead between -1 (B) and 1 (A)
box.set_compartments(np.sin(np.linspace(0, 12, 100)))
box.add_compartment(0, 500_000, "B")                     # or set a block by hand
# box.random_compartments(mean_block_mb=0.8, seed=1)     # or random alternating blocks

cfg = box.config(n_steps=20000, n_lef=6, seed=2)         # config for this region and resolution
geom = ls.geometry(cfg)
inputs = ls.read_inputs(cfg, geom, **box.arrays())       # the sandbox as inputs
mc = ls.run_mc(cfg, inputs, geom)                        # then everything works as above
ls.plots.kymograph(mc, geom, inputs)
```

| Method | What it does |
|---|---|
| `add_ctcf(position, orientation, strength)` | one CTCF motif; orientation is `"forward"`, `"reverse"`, `"unknown"` or a probability of reverse in [0, 1] |
| `add_ctcfs(positions, orientation, strength)` | several motifs; orientation and strength are one value or one per position |
| `add_loop(left, right, weight)` | a fixed loop between two positions, instead of single motifs |
| `set_compartments(values)` | your array, one value per bead in [-1, 1]; other lengths or ranges raise `ValueError` |
| `add_compartment(start, end, state)` | A (`"A"` = 1), B (`"B"` = -1) or a number in [-1, 1] on a block |
| `random_compartments(mean_block_mb, seed)` | alternating A and B blocks of random length |
| `ctcf_sites`, `compartments` | the motifs (table) and the signal (array) defined so far |
| `config(**parameters)` | a `Config` for the region and resolution |
| `arrays()` | the keyword arguments for `ls.read_inputs` |

Use either CTCF motifs or loops in one sandbox, not both.
