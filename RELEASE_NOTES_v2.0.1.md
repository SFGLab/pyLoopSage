# LoopSage v2.0.1

First stable release of LoopSage 2: a full rewrite of the original scripts into an installable package (`pip install pyLoopSage`) with one command that goes from a loop file to an HTML report.

## Highlights

- **One command, one report.** `loopsage -f loops.bedpe -r chr1:77000000-95000000` runs the whole pipeline: input processing, Monte Carlo loop extrusion, 3D molecular dynamics, analysis and an HTML report with contact maps, loop statistics, 3D structures and diagnostics.
- **Proper package.** `src` layout, a `loopsage` command, a Python interface (`import loopsage as ls`) and a documentation site: <https://sfglab.github.io/pyLoopSage/>.
- **Config files or flags.** All parameters live in one `config.ini` (see `config_example.ini`); command-line flags override it. 16 ready-to-run files in `examples/`.

## New since the old scripts

**Input**
- **CTCF motif finder:** give a BEDPE *without* orientation and a genome FASTA (`--motif-genome`). Every anchor is scanned for the CTCF motif (bundled JASPAR MA0139.1 or your own PFM) and orientation probabilities are added automatically. A minimum PET filter (`--min-pet`) is included.
- Reads `.bedpe`, `.bed` / `.narrowPeak` CTCF sites, compartment tracks and `.hic` files (optional Hi-C reference).

**Monte Carlo**
- Faster numba sampler with replicas (`--n-replicas`), multi-threading (`-t`), Metropolis or Annealing, and an optional epigenetic (Potts) compartment model with a free-energy landscape figure.
- New `n_sweep` parameter: loop-extruder moves proposed per MC step (default 2 x number of LEFs).

**Parameter optimizer (`--optimize`)**
- Nelder-Mead search over the number of LEFs, folding, binding and temperature. Fits against Hi-C or the map built from the input loops, with early stopping and re-scoring of the best candidates on longer runs.

**3D / molecular dynamics (OpenMM)**
- Platform chosen automatically: CUDA, then OpenCL, then CPU. Cutoff-based forces make the cost per step grow roughly linearly with chain length.
- Default `sim_step` is now 250 steps per frame, about 20 times faster. Loop-extrusion changes stay visible in the structures; raise `--md-steps` for more relaxation.
- Compartment forces, structure diagnostics, and topology / knot analysis (linking numbers, chain crossings).
- 3D viewer with PNG, MP4 and GIF output.

**Analysis and reporting**
- Validation against Hi-C and experiments, anchor and orientation diagnostics, convergence checks.
- Self-contained HTML report where every figure and table has a plain-language explanation.

**Sensitivity scans**
- New `loopsage-scan` command and `ls.sweep()`: grid, one-at-a-time, random and Sobol designs, run in parallel with a cache and summarised in one report (importance map, Sobol indices, phase-transition diagnostics).

**Usability**
- Console output levels: default shows stages and progress bars, `-v` adds details, `-vv` adds tables and info, `-vvv` debug, `-q` prints nothing.
- Docker image, a test suite, and a quick demo (`python demo/run_demo.py quick`).

## Install

```bash
conda create -n loopsage python=3.11 -y && conda activate loopsage
conda install -c conda-forge "openmm>=8.5.1" pybigwig -y
pip install pyLoopSage
```

Requires Python 3.10 or newer. Full changelog: [CHANGELOG.md](https://github.com/SFGLab/pyLoopSage/blob/main/CHANGELOG.md).
