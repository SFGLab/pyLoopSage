<p align="center">
  <img src="https://raw.githubusercontent.com/SFGLab/pyLoopSage/main/assets/logo.png?v=2" alt="LoopSage" width="560">
</p>

<p align="center">
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://openmm.org"><img alt="OpenMM" src="https://img.shields.io/badge/OpenMM-8.5%2B-6a3de8"></a>
  <a href="https://numba.pydata.org"><img alt="Numba" src="https://img.shields.io/badge/Numba-JIT%20sampler-00A3E0?logo=numba&logoColor=white"></a>
  <a href="https://pyvista.org"><img alt="PyVista" src="https://img.shields.io/badge/PyVista-3D%20viewer-FF7F00"></a>
  <a href="https://www.docker.com"><img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white"></a>
  <a href="https://numpy.org"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white"></a>
  <a href="https://scipy.org"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white"></a>
  <a href="https://mdtraj.org"><img alt="MDTraj" src="https://img.shields.io/badge/MDTraj-trajectories-1a1a1f"></a>
  <a href="https://github.com/SFGLab/pyLoopSage/tree/main/tests"><img alt="pytest" src="https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0.html"><img alt="GPL v3 license" src="https://img.shields.io/badge/License-GPL%20v3-blue"></a>
</p>

**LoopSage** is an energy-based stochastic model of chromatin loop extrusion. Cohesin complexes are modelled as loop-extruding factors (LEFs) that slide along a chromatin fibre, are blocked by CTCF sites and rebind elsewhere. A Metropolis / simulated-annealing Monte Carlo samples ensembles of loop configurations that agree with your loop data (e.g. ChIA-PET, HiChIP, Hi-C loops, or CTCF peaks). Each ensemble can be turned into 3D structures with a short OpenMM molecular-dynamics run.

From one command you get an HTML report with the contact maps, loop statistics, 3D structures and diagnostics.

## Installation

Python 3.10 or newer is required, with **numpy 2.x** (`numpy>=2.0`, installed automatically; numpy 1.x is not supported). Install from [PyPI](https://pypi.org/project/pyLoopSage/) with pip (in a pyenv or any normal Python environment) or with uv. Pick **one** of the options below. A GPU (NVIDIA) is optional and only speeds up the 3D stage.

**pip**

```bash
pip install pyLoopSage                  # CPU / OpenCL
pip install "pyLoopSage[cuda12]"        # NVIDIA GPU, CUDA 12  (or [cuda13] for CUDA 13)
pip install "pyLoopSage[all]"           # + BigWig tracks, .hic maps, topology (see below)
```

**pyenv** (a separate Python version and environment, then pip from PyPI)

```bash
pyenv install 3.11 && pyenv virtualenv 3.11 loopsage
pyenv local loopsage                    # activates it in this folder (or: pyenv activate loopsage)
pip install pyLoopSage                  # add "[cuda12]" for an NVIDIA GPU
```

**uv**

```bash
uv pip install pyLoopSage               # into the active environment (add "[cuda12]" for GPU)
uv add pyLoopSage                       # into a uv project
uvx --from pyLoopSage loopsage -h       # run without installing
```

**From source (development)**

```bash
git clone https://github.com/SFGLab/pyLoopSage.git && cd pyLoopSage
pip install -e ".[all,test]"
```

Optional extras: `bigwig` (BigWig signal tracks, `pyBigWig`), `hic` (`--hic` reference maps, `hic-straw`, which needs a C++ compiler when no wheel exists for your system), `topology` (linking numbers, `topoly`), `all` (the three). Check the installation with `loopsage -h` (version 2.0.1). Details: [Installation](https://sfglab.github.io/pyLoopSage/intro/installation/).

## Usage

```bash
# Minimal run: loops file + region
loopsage -f loops.bedpe -r chr1:77000000-95000000

# Settings from a config file, with command-line overrides
loopsage -c config.ini -t 4 --n-replicas 3

# BEDPE without orientation columns: read CTCF orientation from the reference genome, keep loops with >= 3 PET
loopsage -f loops.bedpe -r chr1:77000000-95000000 --motif-genome hg38.fa --min-pet 3

# See every option
loopsage -h
```

- **Input:** a `.bedpe` loop file, or `.bed` / `.narrowPeak` CTCF sites, plus the genomic region.
- **CTCF motif search (optional):** with `--motif-genome FASTA` every loop anchor is scanned for the CTCF motif (bundled JASPAR MA0139.1, or your own `--motif-pfm`) and the orientation probabilities are added automatically; `--min-pet N` keeps only loops with at least N PET counts (works without a genome too). Needs no extra packages; the genome is read through a `.fai` index and never loaded into memory. Hits are accepted at `--motif-pvalue` (default 1e-5). The report shows a progress bar while searching and a diagnostics section (loops kept, motif scores, orientation calls). The annotated loops are saved in the run's `data` folder.
- **Config:** `config_example.ini` lists every parameter with a short description; command-line flags override it.
- **Output:** the folder given with `--out` (created if missing, e.g. `--out results/chr1_134_489`; default `results/<chrom>_<start>_<end>`), with the report in `report/report.html`.
- **MD platform:** `--platform auto` (default) tries CUDA, then OpenCL, then the multi-threaded CPU platform, and the chosen platform is always printed. Force one with `--platform CUDA|OpenCL|CPU`.
- **Monte Carlo:** `--method Metropolis|Annealing`, `--t-init`, `--t-final`, `--mc-steps`, `--sample-every` and `--burnin` control the sampler (all also keys in the config file).
- **Parallelism:** `-t/--threads` sets worker threads and `--n-replicas` runs independent Monte Carlo chains (default 1).
- **Demo:** `python demo/run_demo.py quick` runs a small example in seconds.

Detailed documentation of the model, parameters and outputs: **[sfglab.github.io/pyLoopSage](https://sfglab.github.io/pyLoopSage/)** (sources in [`markdown/`](https://github.com/SFGLab/pyLoopSage/tree/main/markdown)).

## Use from Python

After `pip install pyLoopSage` the package is imported as `loopsage`. Each step is a function:

```python
import loopsage as ls

cfg = ls.config(region="chr1:77000000-95000000", n_beads=120, n_steps=20000)
geom = ls.geometry(cfg)
inputs = ls.read_inputs(cfg, geom, loops="loops.bedpe")   # inputs are file paths
mc = ls.run_mc(cfg, inputs, geom)                         # Monte Carlo
ls.results.mean_loop_length(mc, geom)                     # results as arrays and tables
md = ls.run_md(cfg, mc, geom)                             # 3D structures

import loopsage.monte_carlo as mc                         # every module can be imported on its own
```

See the [Python interface](https://sfglab.github.io/pyLoopSage/reference/python/) page of the documentation.

## Tests

Run the whole suite in parallel, with a per-module report:

```bash
python tools/run_tests.py             # all tests, one worker per two cores
python tools/run_tests.py -j 8 --html # 8 workers, also writes test_report/report.html
python tools/run_tests.py -k contrast # only tests whose name matches
```

Plain `python -m pytest -q` (after `pip install -e .`) also works, but runs the files one after another (about 15 minutes). Tests of optional packages (`hicstraw`, `pyBigWig`) are skipped when those are not installed.

## Citation

If you use LoopSage, please cite:

- Korsak, S. and Plewczynski, D. *LoopSage: An energy-based Monte Carlo approach for the loop extrusion modelling of chromatin.* Methods (2024).
- Jodkowska, K., Parteka-Tojek, Z., Agarwal, A., Denkiewicz, M., Korsak, S., Chiliński, M., Banecki, K., & Plewczynski, D. *Improved cohesin HiChIP protocol and bioinformatic analysis for robust detection of chromatin loops and stripes.* bioRxiv (2024). https://doi.org/10.1101/2024.05.16.594268

## License

GNU General Public License v3.0.
