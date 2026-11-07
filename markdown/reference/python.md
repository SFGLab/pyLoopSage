# Python interface

After `pip install pyLoopSage`, LoopSage can be used from any Python script, notebook or console. The package on PyPI is called **pyLoopSage**; the name to import is **`loopsage`**.

```bash
pip install pyLoopSage           # add --pre while the release is a release candidate
python -c "import loopsage as ls; print(ls.__version__)"
```

This installs the `loopsage` command and the `loopsage` Python package together. Everything the command does is available from Python, with the same parameters and the same results. The difference is where things come from and go to: in Python, the **inputs can be variables** (arrays, lists, data frames) and the **outputs are returned as variables**, so a simulation can run without reading or writing any file.

## Which way to use it

| You want | Use | Files written |
|---|---|---|
| loops and tracks from variables, results as variables | [`ls.simulate`](#run-a-simulation-and-get-variables), [`ls.simulate_md`](#3d-structures) | none |
| the full pipeline with figures and the HTML report | [`ls.run`, `ls.Simulation`](#the-simulation-class-files-and-report) | a run folder |
| exactly the `loopsage` command | [`ls.main`](#the-command-from-python) or `python -m loopsage` | a run folder |
| one internal piece (a reader, the sampler, a map function) | [`import loopsage.<module>`](#import-any-module) | none |

All four share the same parameters, described next.

## A first example, without files

```python
import numpy as np
import loopsage as ls

# loops as arrays: genomic positions (bp) of the left and right anchors, and a weight per loop
left   = np.array([77_500_000, 80_000_000, 83_000_000, 88_000_000])
right  = left + np.array([3_000_000, 4_000_000, 2_500_000, 5_000_000])
weight = [10, 20, 5, 8]                                   # e.g. PET counts

mc = ls.simulate(loops=dict(left=left, right=right, weight=weight),
                 region="chr1:77000000-95000000", n_beads=120,
                 n_steps=20000, seed=1)                   # the Monte Carlo runs in memory

mc.Ms, mc.Ns                  # left and right leg of every LEF at every snapshot
mc.Es                         # energy at every snapshot
cmap = ls.contact_map(mc)     # 120 x 120 contact map implied by the loops
```

Nothing is written to disk. The pieces are explained in the sections below.

## Parameters as Python variables

Every parameter of a [config file](config.md) is a keyword argument, by its **key name** (`n_steps`, `folding_coeff`, `run_md`). The command-line option can have another name: `--mc-steps` sets `n_steps`, `--md-steps` sets `sim_step`. The tables of the [command line](cli.md) and the [config file](config.md) show both names side by side.

```python
ls.simulate(loops=..., region="chr1:77000000-95000000", n_steps=20000, folding_coeff=1.5, seed=3)
```

**Find a parameter.** `ls.parameters()` returns every parameter as `{name: (default, help)}`, and a search text keeps the matching ones:

```python
ls.parameters("md steps")        # {'sim_step': (100, 'MD steps between two frames ...'), ...}
ls.parameters()["n_steps"][0]    # default of one parameter: 20000
```

**Values.** They are converted like in a config file, so numbers, `True` / `False`, lists and units work: `integrator_step="100 fs"`, `bw_files=["a.bw", "b.bw"]`, `run_md="yes"`. A wrong name or value raises `ValueError` with a clear message (the command line prints a usage message and exits instead), so a script or notebook can catch it.

**Region.** `region="chr1:77000000-95000000"` (in bp) is a shortcut for `chrom`, `region_start` and `region_end`. `n_beads` sets the resolution; without it, one bead is about 2 kb.

**A configuration object.** `ls.config(...)` checks the settings once and returns a `Config` that can be reused and changed:

```python
cfg = ls.config("config.ini", n_steps=5000)         # a config file with some keys overridden
cfg = ls.config(region="chr1:1-9000000", n_beads=100, run_md=False)   # or no file at all
cfg2 = ls.config(cfg, seed=7)                       # a copy with one change

cfg.n_steps                  # every parameter is an attribute
cfg.as_dict()                # all parameters as a dictionary
cfg.source["n_steps"]        # where the value came from: default, ini or cli
ls.write_ini("used.ini", cfg.as_dict(), cfg.source)  # save it as a config file, if you want to
```

Wherever a function takes `config`, it accepts a path to a `.ini` file, a `Config`, or a dictionary. Keyword arguments always win over it. The next example uses all three.

```python
ls.simulate(cfg, loops=...)                         # a Config
ls.simulate("config.ini", loops=..., seed=2)        # a file, with the seed changed
ls.simulate({"n_steps": 3000, "burnin": 500, "region": "chr1:77000000-95000000"}, loops=...)   # a dictionary
```

## Inputs as Python variables

The model needs the loops (or binding sites) and, optionally, extra tracks, all at **bead resolution**. Instead of a BEDPE or BigWig file, give them as arrays. Positions are either genomic coordinates in bp (`units="bp"`, the default) or bead numbers (`units="bead"`). Anything outside the region is ignored.

### Loops

```python
ls.simulate(loops=dict(left=left, right=right, weight=weight), ...)
ls.simulate(loops=(left, right, weight), ...)          # the same, as a tuple
```

| Key | Meaning |
|---|---|
| `left`, `right` | positions of the two anchors of every loop (1D arrays of equal length) |
| `weight` | strength of each loop, such as the number of read pairs (default 1) |
| `left_reverse`, `right_reverse` | probability that the CTCF at each anchor is in the reverse orientation, between 0 and 1 (`-1` = unknown); leave out when unknown, then the weight is split evenly between the left and right roles |
| `units` | `"bp"` or `"bead"` |
| `smooth`, `contrastive`, `loss_sigma` | the same switches as `smoothing_input`, `binding_mode` and `loss_sigma` in the config |

From a data frame, take its columns:

```python
import pandas as pd
df = pd.read_csv("loops.bedpe", sep="\t", header=None)
loops = dict(left=(df[1] + df[2]) // 2, right=(df[4] + df[5]) // 2, weight=df[6])     # anchor centres
```

### Binding sites (peaks) instead of loops

```python
ls.simulate(sites=dict(position=peak_centres, weight=signal, reverse=prob_reverse), ...)
```

Sites give the anchor profiles `L` and `R` only. There are no loops, so `binding_mode = contrastive` has nothing to count.

### Extra tracks

All of these are one value per bead (arrays of length `n_beads`):

```python
ls.simulate(loops=..., tracks=[pol2, h3k27ac], track_names=["Pol2", "H3K27ac"],   # extra binding tracks
            lef_track=nipbl,                                                      # where LEFs prefer to load (>= 0)
            compartments=pc1,                                                     # A/B signal, B negative
            ...)
```

`compartments` is rescaled to [-1, 1] when it is outside that range. To resample a signal that is not on the bead grid, use `geom.bead_of(positions_bp)` to find the bead of each position and average within beads.

### Build the inputs once, reuse them

`ls.simulate` builds the inputs on every call. To keep them, or to compare several settings on the same input, build them first and pass `inputs=`:

```python
geom = ls.geometry("chr1", 77_000_000, 95_000_000, n_beads=120)
inputs = ls.inputs_from_arrays(geom, loops=dict(left=left, right=right, weight=weight), tracks=[pol2])

inputs.loops.L, inputs.loops.R        # anchor profiles per bead (scaled to mean 1)
inputs.loops.cover                    # loop coverage per bead in [-1, 1] (contrastive binding)
inputs.loops.stats                    # number of loops, loop lengths, ...

for f in (0.5, 1.0, 2.0):
    mc = ls.simulate(inputs=inputs, region="chr1:77000000-95000000", n_beads=120, folding_coeff=f)
```

`ls.loops_from_arrays(geom, left, right, ...)` and `ls.sites_from_arrays(geom, position, ...)` build only the loop or site part, as a `LoopData`. You can pass it directly as `loops=`.

## Run a simulation and get variables

```python
mc = ls.simulate(config=None, region=None, loops=..., **parameters)
```

The result is an `MCResult` with these arrays (a *snapshot* is the state saved every `mc_step` steps; there are `n_steps / mc_step` of them):

| Field | Shape | Meaning |
|---|---|---|
| `mc.Ms`, `mc.Ns` | LEFs x snapshots | bead of the left and right leg of every LEF |
| `mc.Es` | snapshots | total energy |
| `mc.Fs`, `mc.Bs`, `mc.Ks` | snapshots | folding, binding and crossing parts of the energy |
| `mc.Ts` | snapshots | temperature (changes with annealing) |
| `mc.ufs` | snapshots | share of the chain that is not inside any loop |
| `mc.epi` | beads x snapshots | epigenetic state of each bead (Potts model) |
| `mc.acc` | 3 x 2 | proposed and accepted moves: rebind, slide, state change |
| `mc.meta` | dictionary | number of beads and LEFs, steps, seed, run time, ... |
| `mc.replicas` | list | the other independent chains when `replicas > 1` (`mc` itself is chain 0) |

The first `burnin` steps are the warm-up; `mc.production(mc.Ms)` returns only the snapshots after it, and `mc.pooled()` joins all chains.

```python
ms, ns = mc.production(mc.Ms), mc.production(mc.Ns)
loop_lengths = np.abs(ns - ms)                     # in beads, for every LEF and snapshot
print(loop_lengths.mean() * geom.bp_per_bead / 1e3, "kb on average")

import matplotlib.pyplot as plt
plt.plot(mc.Es); plt.xlabel("snapshot"); plt.ylabel("energy")       # did it settle?
plt.imshow(ls.contact_map(mc), cmap="Reds"); plt.show()
```

`ls.contact_map(mc)` is the contact map implied by the loops (production snapshots of all chains), as an `n_beads x n_beads` array. The same seed gives the same result, and `seed=None` (the default) draws a random one, stored in `mc.meta["seed"]`. The first call compiles the sampler (a few seconds) and later calls reuse it.

Save and load a result as a file only if you want to: `mc.save("mc.npz")` and `ls.monte_carlo.simulation.MCResult.load("mc.npz")`.

## 3D structures

```python
md = ls.simulate_md(mc, region="chr1:77000000-95000000", n_beads=120, sim_step=500)
```

It takes the loops of the snapshots in `mc` and builds one 3D structure per snapshot with OpenMM. The settings are the usual `molecular_dynamics` parameters (`sim_step` = steps per frame, `platform`, `ev_ff_strength`, ...). The result is an `MDResult`:

| Field | Shape | Meaning |
|---|---|---|
| `md.coords` | frames x beads x 3 | positions in nm |
| `md.avg_map`, `md.std_map` | beads x beads | mean and spread of the contact strength over the frames |
| `md.avg_dist`, `md.std_dist` | beads x beads | mean and spread of the pairwise distances (nm) |
| `md.rg` | frames | radius of gyration (nm) |
| `md.epot` | frames | potential energy (kJ/mol) |
| `md.frame_idx` | frames | index of the Monte Carlo snapshot each frame came from |

OpenMM needs a few scratch files; they are put in a temporary folder that is deleted when the function returns, so nothing is left behind. The settings that must match the Monte Carlo run (region, `n_beads`) are best taken from one shared dictionary:

```python
kw = dict(region="chr1:77000000-95000000", n_beads=120, n_steps=20000, seed=1)
mc = ls.simulate(loops=loops, **kw)
md = ls.simulate_md(mc, **kw, sim_step=500)
```

## The `Simulation` class: files and report

When you want what the command produces (run folder, figures, `report.html`), use `ls.Simulation`. It takes the same `config`, `region` and parameters:

```python
sim = ls.Simulation("config.ini", n_steps=20000)      # or interaction_file="loops.bedpe", region=..., ...
sim.run()                                             # every enabled stage and the report
ls.run("config.ini", n_steps=20000)                   # the same in one call
```

The pipeline can also run **one step at a time**. A step first runs the steps it needs and keeps their results, so nothing is computed twice:

| Method | What it does | Result (also `sim.<name>`) |
|---|---|---|
| `sim.preprocess()` | read the loops, tracks and optional motif search onto the bead grid | `inputs` |
| `sim.optimize()` | fit the number of LEFs, $f$, $b$ and the temperature (Nelder&ndash;Mead) | `optimization` |
| `sim.monte_carlo()` | loop extrusion Monte Carlo | `mc` |
| `sim.molecular_dynamics()` | 3D structures | `md` |
| `sim.structure()` | 3D structure analysis and viewer | `structure` |
| `sim.analyze()` | compare the simulated maps with the experiment | `validation`, `features` |
| `sim.links()` | linking numbers of loops (needs `topoly`) | `links` |
| `sim.report()` | write `report.html` | path of the report |

`sim.run_dir` is the folder of the run, `sim.cfg` the configuration used. A `Simulation` writes into one numbered run folder (like the command); call `run()` only on a fresh one.

**With inputs as variables.** `Simulation(loops=..., sites=..., tracks=..., inputs=...)` takes the same arguments as `ls.simulate`. Then `preprocess()`, `monte_carlo()` and `molecular_dynamics()` are available and return their results, while `optimize`, `analyze`, `links`, `report` and `run` raise a `ValueError`, because they compare with the loops *file*. They need `interaction_file`. If you only want the arrays back and no run folder, use `ls.simulate` and `ls.simulate_md`.

## The command from Python

```python
ls.main(["-c", "config.ini", "-vv"])        # exactly `loopsage -c config.ini -vv`
```

`python -m loopsage -c config.ini` does the same from a shell. `main` takes the list of command-line words and runs the whole pipeline. Mistakes in the options print the usual usage message and exit.

## Import any module

Every part of LoopSage is importable under the name `loopsage`, for example `import loopsage.monte_carlo as mc` or `from loopsage.preproc.loader import load_inputs`:

| Package | Contents |
|---|---|
| `loopsage.settings` | the parameters, config files, `make_config` and `load_config` |
| `loopsage.preproc` | reading loops, peaks, tracks and `.hic` files; `arrays` builds inputs from variables |
| `loopsage.motif_finder` | CTCF motif search and orientation |
| `loopsage.monte_carlo` | energies, moves and the sampler (`simulation.run_mc`) |
| `loopsage.optimizer` | loss and Nelder&ndash;Mead search |
| `loopsage.molecular_dynamics` | OpenMM 3D simulation |
| `loopsage.analysis`, `loopsage.validation` | maps, features and the comparison with experiment |
| `loopsage.knots` | linking numbers |
| `loopsage.viz`, `loopsage.viz3d`, `loopsage.reporting` | figures, the viewer and the report |
| `loopsage.pipeline` | the stages and the runner |

For example, the sampler can be called directly with the objects it works on:

```python
from loopsage.settings.loader import make_config
from loopsage.preproc.arrays import geometry, inputs_from_arrays
from loopsage.monte_carlo.simulation import run_mc
from loopsage.analysis.maps import loop_map

cfg = make_config(region="chr1:77000000-95000000", n_beads=120, n_steps=20000, seed=1)
geom = geometry("chr1", 77_000_000, 95_000_000, 120)
inputs = inputs_from_arrays(geom, loops=(left, right, weight))
mc = run_mc(cfg, inputs, geom, quiet=True)
cmap = loop_map(mc, geom.n_beads)
```

`import loopsage` itself is fast: the heavy modules (Numba, OpenMM, plotting) load when you first use them. The names are the same objects as in the source tree, so `loopsage.monte_carlo` and `monte_carlo` are one module, not two copies.

## Recipes

**A parameter scan in memory.**

```python
results = {}
for f in (0.5, 1.0, 2.0):
    mc = ls.simulate(inputs=inputs, region="chr1:77000000-95000000", n_beads=120, folding_coeff=f, seed=1)
    results[f] = ls.contact_map(mc)
```

**Several independent chains** for convergence checks: `replicas=4` gives `mc.replicas`, and `mc.chains` lists all four.

**Reproducible runs.** Set `seed`. Two calls with the same inputs, parameters and seed return identical arrays.

**Less output.** `quiet=True` removes the progress bars and messages; `verbosity=2` (or `-vv` on the command line) shows more.

**Notebooks.** The functions work in Jupyter as they are. If a run is interrupted, nothing is left half-written, because `ls.simulate` writes no files.

## Function reference

The most important functions of each part. Every function below is available as `ls.<name>` after `import loopsage as ls`. Types: *array* means a NumPy array or anything NumPy can convert (list, pandas column). *config* is explained in [Parameters as Python variables](#parameters-as-python-variables): an `.ini` path, a `Config` or a dictionary. `**parameters` are any [config keys](config.md) by name.

### `ls.simulate`

```python
ls.simulate(config=None, region=None, *, loops=None, sites=None, tracks=None, track_names=None,
            lef_track=None, compartments=None, inputs=None, **parameters) -> MCResult
```

Runs the loop extrusion Monte Carlo entirely in memory. No folder, file or figure is written.

| Input | Meaning |
|---|---|
| `config` | settings from an `.ini` file, a `Config` or a dictionary (optional) |
| `region` | `"chr1:77000000-95000000"` in bp; or give `chrom`, `region_start`, `region_end` |
| `loops` | the loops, as `dict(left=..., right=..., weight=...)` or the tuple `(left, right[, weight])`; see [Loops](#loops) |
| `sites` | binding sites instead of loops: `dict(position=..., weight=..., reverse=...)` |
| `tracks`, `track_names` | extra binding tracks, array `(n_tracks, n_beads)` and their names |
| `lef_track` | per-bead preference for loading a LEF (array, `n_beads`) |
| `compartments` | per-bead A/B signal (array, `n_beads`; B negative) |
| `inputs` | ready inputs from `ls.inputs_from_arrays`, instead of `loops`, `sites`, ... |
| `**parameters` | any config key, e.g. `n_beads`, `n_steps`, `mc_step`, `burnin`, `n_lef`, `folding_coeff`, `bind_coeff`, `seed`, `replicas` |

**Returns** an [`MCResult`](#mcresult). Without `loops`, `sites` or `inputs`, the input file of the configuration (`interaction_file`) is read.

```python
mc = ls.simulate(loops=(left, right, weight), region="chr1:77000000-95000000",
                 n_beads=120, n_steps=20000, mc_step=100, burnin=4000, replicas=2, seed=1)
print(mc.n_samples, mc.meta["n_lef"])           # 200 snapshots, number of LEFs
```

### `ls.simulate_md`

```python
ls.simulate_md(mc, config=None, region=None, **parameters) -> MDResult
```

Builds a 3D structure for every snapshot of a Monte Carlo result with OpenMM. Scratch files go to a temporary folder that is deleted afterwards.

| Input | Meaning |
|---|---|
| `mc` | the `MCResult` from `ls.simulate` |
| `config`, `region`, `**parameters` | as for `ls.simulate`; `region` and `n_beads` must be the same as for `mc` |

Useful parameters: `sim_step` (MD steps per frame), `platform` (`CPU`, `CUDA`, `OpenCL`), `ev_ff_strength` and `ev_ff_power` (excluded volume), `md_skip_burnin`, `threads`.

**Returns** an [`MDResult`](#mdresult).

```python
md = ls.simulate_md(mc, region="chr1:77000000-95000000", n_beads=120, sim_step=500, platform="CPU")
md.coords.shape            # (frames, 120, 3)
```

### `ls.contact_map`

```python
ls.contact_map(mc, n_beads=None, **options) -> ndarray
```

Contact map implied by the loops of `mc` (production snapshots of all chains).

| Input | Meaning |
|---|---|
| `mc` | an `MCResult` |
| `n_beads` | size of the map (default: the number of beads of `mc`) |
| `**options` | map shape: `dot`, `domain`, `hop`, `depth`, `sigma`, `base` (see `loopsage.analysis.maps.loop_map`) |

**Returns** a symmetric array `(n_beads, n_beads)`, scaled so that a loop is about 1.

```python
cmap = ls.contact_map(mc)
plt.imshow(np.log(cmap + 1e-3), cmap="Reds")
```

### `ls.geometry`

```python
ls.geometry(chrom, start, end, n_beads=None) -> Geometry
```

The region and the bead grid. `start`, `end` in bp; `n_beads` defaults to about one bead per 2 kb.

A `Geometry` has the fields `chrom`, `start`, `end`, `n_beads` and:

| Member | Meaning |
|---|---|
| `bp_per_bead` | bases in one bead |
| `bead_of(bp)` | bead index of one or many genomic positions (clipped to the region) |
| `beads_to_bp(n)` | beads to bp |
| `axis_mb()` | centre of every bead in Mb, for plot axes |
| `label` | text such as `chr1:77,000,000-95,000,000` |

```python
geom = ls.geometry("chr1", 77_000_000, 95_000_000, 120)
geom.bp_per_bead                      # 150000
geom.bead_of([77_500_000, 90_000_000])    # array([ 3, 86])
```

### `ls.inputs_from_arrays`

```python
ls.inputs_from_arrays(geom, loops=None, sites=None, tracks=None, track_names=None,
                      lef_track=None, compartments=None, **options) -> InputData
```

Builds all model inputs from variables.

| Input | Meaning |
|---|---|
| `geom` | a `Geometry` |
| `loops` | `dict(left, right, weight, left_reverse, right_reverse, units)`, a tuple `(left, right[, weight])`, or a `LoopData` |
| `sites` | used when there are no loops: `dict(position, weight, reverse, units)` |
| `tracks` | extra tracks `(n_tracks, n_beads)`; `track_names` names them |
| `lef_track`, `compartments` | per-bead arrays, see [Extra tracks](#extra-tracks) |
| `**options` | `smooth`, `contrastive`, `sigma`, `loss_sigma`, `units`, passed to the loop builder |

**Returns** an [`InputData`](#inputdata). Raises `ValueError` for arrays of the wrong length, probabilities outside [0, 1], or when neither loops nor sites are given.

```python
inputs = ls.inputs_from_arrays(geom, loops=dict(left=left, right=right, weight=weight), tracks=[pol2])
```

### `ls.config` and `ls.parameters`

```python
ls.config(config=None, region=None, **parameters) -> Config
ls.parameters(search=None) -> dict
```

`ls.config` merges defaults, the optional `config`, and the keyword arguments (they win), converts the values, checks them, and returns a `Config`. It raises `ValueError` on an unknown key or a bad value.

A `Config` has every parameter as an attribute (`cfg.n_steps`), `cfg.as_dict()` for all of them, `cfg.source[key]` (`default`, `ini` or `cli`), and `cfg.notes` for warnings such as ignored keys.

`ls.parameters(search)` returns `{name: (default, help)}` for all parameters whose name or help contains `search` (all when `None`).

```python
cfg = ls.config("config.ini", n_steps=5000)
ls.parameters("temperature")       # {'t_init': (...), 't_final': (...), 'sim_temp': (...), ...}
```

### `ls.Simulation`, `ls.run` and `ls.main`

```python
ls.Simulation(config=None, region=None, *, inputs=None, loops=None, sites=None, tracks=None,
              track_names=None, lef_track=None, compartments=None, **parameters)
ls.run(config=None, region=None, **parameters) -> Simulation
ls.main(argv=None)
```

`Simulation` is one run with its folder, figures and report. `ls.run(...)` is `Simulation(...).run()`; `ls.main(argv)` is the command (it takes the list of command-line words).

| Member | Meaning |
|---|---|
| `sim.run()` | all enabled stages and the report; returns `sim` |
| `sim.preprocess()` ... `sim.report()` | single steps, see [the table above](#the-simulation-class-files-and-report) |
| `sim.cfg` | the `Config` used |
| `sim.run_dir` | the run folder |
| `sim.geometry` | the `Geometry` of the run |
| `sim.inputs`, `sim.mc`, `sim.md`, `sim.optimization`, `sim.validation`, `sim.features`, `sim.links`, `sim.report_path` | results of the steps that have run |

```python
sim = ls.Simulation("config.ini", n_steps=20000, run_md=False)
mc = sim.monte_carlo()
print(sim.run_dir, mc.n_samples)
sim.analyze(); sim.report()          # compare with the experiment and write report.html
```

### Result and input objects

#### `MCResult`

Returned by `ls.simulate` and `sim.monte_carlo()`. A *snapshot* is a state saved every `mc_step` steps.

| Member | Meaning |
|---|---|
| `Ms`, `Ns` | `(LEFs, snapshots)` left and right leg bead of every LEF |
| `Es`, `Fs`, `Bs`, `Ks` | `(snapshots)` total, folding, binding and crossing energy |
| `Ts` | `(snapshots)` temperature |
| `ufs` | `(snapshots)` share of the chain outside all loops |
| `epi` | `(beads, snapshots)` epigenetic state (Potts model) |
| `acc` | `(3, 2)` proposed and accepted moves: rebind, slide, state change |
| `meta` | dictionary: `n_beads`, `n_lef`, `n_steps`, `mc_step`, `burnin`, `seed`, `runtime_s`, ... |
| `n_samples` | number of snapshots |
| `production(a)` | the part of an array `a` (last axis) after the burn-in |
| `pooled()` | one `MCResult` with the production snapshots of all chains |
| `chains`, `replicas` | all independent chains / the ones besides this one |
| `save(path)`, `MCResult.load(path)` | optional `.npz` file |

#### `MDResult`

Returned by `ls.simulate_md` and `sim.molecular_dynamics()`.

| Member | Meaning |
|---|---|
| `coords` | `(frames, beads, 3)` positions in nm |
| `avg_map`, `std_map` | `(beads, beads)` mean and spread of the contact strength |
| `avg_dist`, `std_dist` | `(beads, beads)` mean and spread of distances in nm |
| `rg` | `(frames)` radius of gyration in nm |
| `epot` | `(frames)` potential energy in kJ/mol |
| `frame_idx` | `(frames)` Monte Carlo snapshot of each frame |
| `meta` | dictionary: platform, run time, time step, steps per frame, ... |
| `save(path)`, `MDResult.load(path)` | optional `.npz` file |

#### `InputData` and `LoopData`

Returned by `ls.inputs_from_arrays` and `sim.preprocess()`. Besides the inputs of the model they let you inspect what was read.

| Member | Meaning |
|---|---|
| `inputs.loops` | a `LoopData` with the loops |
| `inputs.BWs`, `inputs.bw_names` | extra tracks `(n_tracks, n_beads)` and their names |
| `inputs.lef_track`, `inputs.comp` | LEF loading preference and A/B signal per bead, or `None` |
| `loops.L`, `loops.R` | left and right anchor profiles per bead (mean 1) |
| `loops.cover` | loop coverage per bead in [-1, 1] (contrastive binding), or `None` |
| `loops.stats` | number of loops and loop-length statistics |

`ls.loops_from_arrays(geom, left, right, weight, ...)` and `ls.sites_from_arrays(geom, position, weight, ...)` build only the `LoopData` part, with the same arguments as the `loops` and `sites` dictionaries.
