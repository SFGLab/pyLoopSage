# Installation

## Requirements

- **Python 3.10 or newer**.
- A Linux, macOS or Windows (WSL) computer. A GPU is optional: it only speeds up the 3D molecular dynamics stage.
- About 2 GB of free disk space for the packages.

## Step by step (recommended: conda)

`openmm` (the molecular dynamics engine) and `pyBigWig` install most reliably from conda, so the packages are installed in two parts. `hic-straw`, which reads `.hic` files for the optional `--hic` reference, is installed with the package; it builds from source when no wheel exists for your system and then needs a C++ compiler and libcurl.

```bash
# 1. get the code
git clone https://github.com/SFGLab/pyLoopSage.git
cd pyLoopSage

# 2. create an isolated environment
conda create -n loopsage python=3.11 -y
conda activate loopsage

# 3. packages that are best installed from conda
conda install -c conda-forge "openmm>=8.5.1" pybigwig -y

# 4. all other packages
pip install -r requirements.txt

# 5. install pyLoopSage itself (creates the `loopsage` command)
pip install -e .
```

Check the installation with `loopsage -h`. The same command also runs as `python -m loopsage`.

## Check that it works

Run the small demo (under a minute) and open the report it writes:

```bash
python demo/run_demo.py quick
```

The demo uses a small synthetic loop file shipped with the repository. It needs no downloads.

To run the automatic tests (several minutes):

```bash
python -m pytest
```

## Choosing the 3D computing platform

The 3D stage runs on the fastest platform it finds, in this order: **CUDA** (NVIDIA GPU), **OpenCL**, then **CPU** with all cores. The chosen platform is always printed. To force one, use `--platform CUDA`, `--platform OpenCL` or `--platform CPU`.

## Docker

A `Dockerfile` with CUDA support is in the repository root:

```bash
docker build -t loopsage .
docker run --gpus all -it -v "$PWD":/app loopsage
```

Inside the container, run LoopSage as described above.

## Optional: CTCF motif search

To let LoopSage read CTCF orientation from a reference genome (`--motif-genome`), you only need a genome FASTA file, for example `hg38.fa`. No extra package is needed; the motif table (JASPAR MA0139.1) is included. See the [CLI reference](../reference/cli.md).

## Building these notes

The documentation is written in Markdown in the `markdown/` folder and built with MkDocs:

```bash
pip install mkdocs-material
mkdocs serve          # preview at http://127.0.0.1:8000
mkdocs build          # write the site to site/
```
