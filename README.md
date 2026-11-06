<p align="center">
  <img src="assets/logo.svg" alt="LoopSage" width="760">
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
  <a href="tests"><img alt="pytest" src="https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0.html"><img alt="GPL v3 license" src="https://img.shields.io/badge/License-GPL%20v3-blue"></a>
</p>

**LoopSage** is an energy-based stochastic model of chromatin loop extrusion. Cohesin complexes are modelled as loop-extruding factors (LEFs) that slide along a chromatin fibre, are blocked by CTCF sites and rebind elsewhere. A Metropolis / simulated-annealing Monte Carlo samples ensembles of loop configurations that agree with your loop data (e.g. ChIA-PET, HiChIP, Hi-C loops, or CTCF peaks). Each ensemble can be turned into 3D structures with a short OpenMM molecular-dynamics run.

From one command you get an HTML report with the contact maps, loop statistics, 3D structures and diagnostics.

## Installation

Python 3.10 or newer is required. `openmm` and `pyBigWig` install most reliably from conda. `hic-straw` (reads `.hic` files for the optional `--hic` reference) is installed with the package. It needs a C++ compiler and libcurl when no wheel exists for your system.

```bash
git clone https://github.com/SFGLab/pyLoopSage.git
cd pyLoopSage
conda create -n loopsage python=3.11 -y && conda activate loopsage
conda install -c conda-forge "openmm>=8.5.1" pybigwig -y
pip install -r requirements.txt && pip install -e .
```

Then install the package (`pip install -e .`) to get the `loopsage` command (version 2.0.0rc2).

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
- **Output:** `results/<chrom>_<start>_<end>_<run>/`, with the report in `report/report.html`.
- **MD platform:** `--platform auto` (default) tries CUDA, then OpenCL, then the multi-threaded CPU platform, and the chosen platform is always printed. Force one with `--platform CUDA|OpenCL|CPU`.
- **Monte Carlo:** `--method Metropolis|Annealing`, `--t-init`, `--t-final`, `--mc-steps`, `--sample-every` and `--burnin` control the sampler (all also keys in the config file).
- **Parallelism:** `-t/--threads` sets worker threads and `--n-replicas` runs independent Monte Carlo chains (default 1).
- **Demo:** `python demo/run_demo.py quick` runs a small example in seconds.

Detailed documentation of the model, parameters and outputs is in [`markdown/`](markdown/).

## Tests

Run the whole suite in parallel, with a per-module report:

```bash
python tools/run_tests.py             # all tests, one worker per two cores
python tools/run_tests.py -j 8 --html # 8 workers, also writes test_report/report.html
python tools/run_tests.py -k contrast # only tests whose name matches
```

Plain `PYTHONPATH=src python -m pytest -q` also works, but runs the files one after another (about 15 minutes). Tests of optional packages (`hicstraw`, `pyBigWig`) are skipped when those are not installed.

## Citation

If you use LoopSage, please cite:

- Korsak, S. and Plewczynski, D. *LoopSage: An energy-based Monte Carlo approach for the loop extrusion modelling of chromatin.* Methods (2024).
- Jodkowska, K., Parteka-Tojek, Z., Agarwal, A., Denkiewicz, M., Korsak, S., Chiliński, M., Banecki, K., & Plewczynski, D. *Improved cohesin HiChIP protocol and bioinformatic analysis for robust detection of chromatin loops and stripes.* bioRxiv (2024). https://doi.org/10.1101/2024.05.16.594268

## License

GNU General Public License v3.0.
