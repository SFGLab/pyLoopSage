# Sensitivity scans

`loopsage-scan` runs many small simulations on a synthetic polymer, varies the parameters you choose, and combines everything into one report, `sensitivity_report.html`. It answers questions such as: *which parameter controls the loop size? where does the system change its behaviour? do links appear in 3D when topoisomerase is slow?*

## Why a sensitivity scan gives biophysical insight

A single simulation shows what one parameter set does. It does not tell you **which parameters matter**, **how strongly**, or **where the behaviour changes**. A scan does, and this is the same thing an experiment does when it varies one condition at a time.

- **Which knob controls which observable.** The *importance map* shows the share of the variance of every output (mean loop size, number of parent and child loops, bound fraction, ...) explained by every input. A knob with a small share can be fixed; a knob with a large share must be measured or fitted.
- **Scaling laws.** The *elasticity* is the percent change of an output per percent change of an input (the log-log slope). An elasticity of 1 means proportional, for example loop size growing with extruder processivity.
- **Interactions.** Sobol indices separate the effect of an input alone (S1) from its effect together with others (ST). ST much larger than S1 means the inputs act in combination.
- **Phase transitions and bifurcations.** Loop extrusion competes with thermal noise. At low temperature loops freeze, at high temperature they dissolve, and in between the system may change abruptly. A dense scan along one input finds where, and tells a smooth change from a sharp one (see below).
- **Topology.** In 3D, loops can link like rings of a chain, and a chain can pass through itself. The scan counts both and shows how they depend on topoisomerase activity and excluded volume.
- **Trust.** Every point is repeated with several seeds, so the report shows noise, equilibration and effective sample size next to every result.

## Quick start

```bash
loopsage-scan --init scan.ini      # write a template
loopsage-scan scan.ini --dry-run   # count the runs and estimate the time
loopsage-scan scan.ini             # run; writes results/scan_<name>/sensitivity_report.html
```

Open the report in any browser; it is one self-contained file. Finished runs are cached, so re-running a scan after you add a value only simulates the new points, and `--replot` rebuilds the report without running anything.

## The scan file

A scan file is an `.ini` file with five sections.

```ini
[scan]                     # how to scan
name       = temperature_and_rings
design     = grid          # grid | oat | random | sobol
replicates = 4             # seeds per point; every point uses the same seeds
snapshots  = 200           # saved configurations per run
metrics    = default       # default | all | a list

[system]                   # the synthetic polymer
n_beads     = 300          # 200 to 500 is typical
bp_per_bead = 10000
n_ctcf      = 12           # CTCF sites
layout      = even         # even | random
orientation = convergent   # convergent | random | forward | reverse

[base]                     # fixed parameters: any key of config.ini
n_steps = 100000

[sweep]                    # what to vary
T     = lin(1, 16, 12)     # 12 temperatures from 1 to 16
n_lef = 4, 12              # two numbers of extruders
```

**Writing an axis.** Each line of `[sweep]` is one input to vary. Values can be a list (`4, 8, 16`), a range (`4:16:4` means 4, 8, 12, 16), `lin(a, b, n)` for `n` evenly spaced values, or `log(a, b, n)` for `n` values evenly spaced on a logarithmic scale. `T` is a shortcut that sets both `t_init` and `t_final`. Polymer settings are scanned as `system.n_beads`. Any key of the [config page](config.md) can be an axis.

**Choosing a design.**

| Design | Runs | Use it to |
|---|---|---|
| `grid` | product of all axes | see every combination, draw heat maps and find transitions |
| `oat` | one axis at a time | get a first local picture with few runs |
| `random` | `samples` quasi-random points | cover many inputs with few runs |
| `sobol` | `samples` x (inputs + 2) | measure S1 and ST with confidence intervals |

The total is printed first by `--dry-run`, with a time estimate; `max_runs` stops accidental explosions.

**Defaults of the sandbox.** Unless you change them in `[base]`, a scan uses the Metropolis sampler at `T = 5` and `bind_coeff = 0.05`. This is a lively regime for a handful of sites, not a recommendation for real data.

## Outputs of the report

The report has these sections: *Overview*, *At a glance* (plain-language findings and the importance map), *Phase transitions and bifurcations*, one section per group of outputs (loops, nesting, binding, energy, 3D structure, topology), *Quality*, *Setup* and a *Glossary*. Next to it the scan writes `sensitivity_runs.csv` (one row per run), `sensitivity_points.csv`, `sensitivity_importance.csv` and, for Sobol designs, `sensitivity_sobol.csv`.

Outputs (metrics) include the mean loop length, unfolded fraction, number of **parent** loops (hold other loops), **child** loops (sit inside another), isolated loops, nesting depth, bound fraction at CTCF sites, anchored-loop fraction and energy. `loopsage-scan --list-metrics` lists all of them.

## Phase transitions and bifurcations

Give one input **at least four values, preferably ten or more**, and three or more replicates. The report then follows each output along that input and looks for a sharp change. For every detected change it shows:

- the **order parameter** (the output) against the **control parameter** (the input), with the transition interval shaded;
- its **fluctuations** in time (susceptibility), which peak at a continuous transition;
- the **autocorrelation time**, which peaks at *critical slowing down*;
- a **bifurcation diagram**: snapshots of the output at each input value, with one line per replicate. Two separate clouds at one value, or replicates that follow different lines, suggest two coexisting states (a first-order transition or a bifurcation);
- **histograms** of the output across the transition;
- with two inputs on a grid, a **phase map**: the transition line as a function of the second input;
- if you scan `system.n_beads` together with `system.ctcf_spacing` (beads between sites, keeps the density fixed), a **Binder cumulant** plot whose crossing marks a continuous transition.

The text next to each case states what the numbers suggest and suggests a finer range, for example `T = lin(1.8, 4.3, 12)`, to resolve it. The scan sees only the grid you give it: a steep crossover and a true transition look alike on a coarse grid, and hysteresis (different paths up and down) cannot be tested because each run starts from scratch. Read "possible bistability" as a hint to confirm with more replicates.

## 3D structure and topology

Add `md = true` (or `--md`) to build a short 3D trajectory in every run and report the radius of gyration, compaction, contact-decay exponent and anchor contacts. Add `topology = true` (or `--topology`) to also count **links** between loops and **chain crossings**:

```ini
[scan]
md       = true
topology = true
md_frames = 20             # 3D frames analysed per run

[sweep]
n_lef     = 8, 16
topo_rate = 0.0, 0.01, 0.05   # topoisomerase events
```

Linked pairs are rare events, so the share of linked loop pairs is pooled over all frames and replicates, with a binomial error bar, rather than averaged run by run. The report also relates **crossing leaks** (the chain passing through itself even though no topoisomerase acted) to the **strand-crossing barrier** of the excluded volume. Sweeping a topology setting (`topo_*`, `ev_*`) switches 3D and topology on automatically. 3D runs are much slower: use fewer points and check `--dry-run`.

## `loopsage-scan` options

| Option | Meaning |
|---|---|
| `scan` | the scan file |
| `-o, --out DIR` | output folder (default `results/scan_<name>`) |
| `-j, --workers N` | worker processes (default: all cores; every worker uses one thread) |
| `--set SECTION.KEY=VALUE` | override a setting, e.g. `--set scan.replicates=8 --set sweep.T=1,2,4` |
| `--md`, `--topology` | switch on 3D structures, and link and crossing analysis |
| `--dry-run` | count the runs and estimate the time; run nothing |
| `--replot` | rebuild the report from the cache |
| `--no-cache` | ignore the cache of finished runs |
| `--init FILE` | write a template scan file |
| `--list-metrics` | list all outputs |
| `-q, --quiet` | no progress bar |

## From Python

```python
import loopsage as ls

res = ls.sweep({"T": "lin(1, 16, 12)", "n_lef": [4, 12]},
               system={"n_beads": 300, "n_ctcf": 12}, base={"n_steps": 100000},
               replicates=4, out="my_scan")       # writes my_scan/sensitivity_report.html
res.table                                         # one row per run (pandas DataFrame)
```

`res.table` has the inputs, the mean of every output over the production part of each run, and, for each output, its autocorrelation time (`tau:<name>`), its fluctuation (`std:<name>`) and its Binder cumulant (`u4:<name>`).

## Notes on running time and accuracy

Runs are independent, so they are spread over all cores, and each run gives up its raw trajectory as soon as the outputs are computed, so memory stays small. The same seeds are used at every point (common random numbers), which makes differences between points much less noisy than the noise of single runs. More snapshots per run do not add independent information; more steps or more replicates do (the *Quality* section tells you which is needed).
