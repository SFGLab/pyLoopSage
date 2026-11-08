# Reference

Complete lists of everything you can set when running LoopSage, and the three ways to run it.

## Usage

LoopSage can be used in three ways. They run the same model with the same parameters, so you can start with the one that suits you and change later.

### Command line

Everything is given as options. Good for a quick run or a script.

```bash
loopsage -f loops.bedpe -r chr1:77000000-95000000 --mc-steps 20000
```

The results (figures, report, structures) are written to a run folder, `results/chr1_77000000_95000000_1/`. All options are listed on the [command-line page](cli.md).

### Config file

The settings are written in a text file, which is easy to keep, share and repeat. The command-line options still work and win over the file.

```ini
# config.ini
[input]
interaction_file = loops.bedpe
chrom = chr1
region_start = 77000000
region_end = 95000000

[monte_carlo]
n_steps = 20000
```

```bash
loopsage -c config.ini
```

The run writes the same run folder as before, including `config_resolved.ini` with every setting that was used. All keys are listed on the [config page](config.md).

### Python

Each step is a function and the results are Python objects, so you can work in a notebook or script without writing files. Inputs are given as file paths.

```python
import loopsage as ls

cfg = ls.config(region="chr1:77000000-95000000", n_beads=120, n_steps=20000)
geom = ls.geometry(cfg)
inputs = ls.read_inputs(cfg, geom, loops="loops.bedpe")
mc = ls.run_mc(cfg, inputs, geom)                        # Monte Carlo loop extrusion
ls.results.mean_loop_length(mc, geom)                    # a result as an array
```

The functions, the results and the sandbox are described on the [Python page](python.md).

## Pages

| Page | What it covers |
|---|---|
| [Command-line interface](cli.md) | The `loopsage` command, its general options and every option by topic, with defaults. |
| [Python interface](python.md) | Installing with `pip` and calling LoopSage from Python (`import loopsage as ls`): one function per step, results as arrays and figures, and a sandbox for synthetic systems. |
| [Config files](config.md) | How to write a `.ini` config file, and a table of every key (including the advanced ones that have no command-line option). |

The command-line and config pages are generated from the same parameter list in the code, so they always agree with the program.
