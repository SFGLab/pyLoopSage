# Installation

## Requirements

- **Python 3.10 or newer**.
- **numpy 2.x** (`numpy>=2.0`; `numba>=0.60` for numpy 2 support). Both are installed automatically. numpy 1.x is not supported, because numpy 1 and 2 are not binary compatible. LoopSage is tested with numpy 2.4 and 2.5.
- A Linux, macOS or Windows (WSL) computer. A GPU is optional: it only speeds up the 3D molecular dynamics stage.
- About 2 GB of free disk space for the packages.

## Choose an installation method

| Method | Command | Good for |
|---|---|---|
| **pip** | `pip install pyLoopSage` | most users |
| **pyenv** | `pyenv virtualenv 3.11 loopsage` then `pip install pyLoopSage` | several Python versions on one machine |
| **uv** | `uv pip install pyLoopSage` | fast installs, project environments |
| **source** | `pip install -e ".[all,test]"` | development |

All methods install the same core packages: numpy 2.x, scipy, pandas, numba, matplotlib, rich, OpenMM, pyvista, imageio and pillow. Nothing else is needed to run the pipeline.

### pip

```bash
python -m venv loopsage-env && source loopsage-env/bin/activate
pip install pyLoopSage
loopsage -h
```

### pyenv

[pyenv](https://github.com/pyenv/pyenv) installs and switches between Python versions. With the `pyenv-virtualenv` plugin each version gets its own environment, and the package comes from [PyPI](https://pypi.org/project/pyLoopSage/):

```bash
pyenv install 3.11
pyenv virtualenv 3.11 loopsage
pyenv local loopsage            # activates the environment inside the current folder
pip install pyLoopSage          # or pip install "pyLoopSage[cuda12]" for an NVIDIA GPU
loopsage -h
```

### uv

[uv](https://docs.astral.sh/uv/) is a fast installer for Python packages. It reads the same package from PyPI:

```bash
uv venv && source .venv/bin/activate
uv pip install pyLoopSage          # or: uv add pyLoopSage   (inside a uv project)
uvx --from pyLoopSage loopsage -h  # one-off run without installing
```

### From source

```bash
git clone https://github.com/SFGLab/pyLoopSage.git
cd pyLoopSage
pip install -e ".[all,test]"
```

With uv, `uv sync --extra all --extra test` installs exactly the package versions pinned in `uv.lock`, so every developer and every CI run gets the same environment. After you change the dependencies in `pyproject.toml`, run `uv lock` and commit the updated `uv.lock`. The lock file is for development only: users who install from PyPI get the versions allowed by `pyproject.toml`.

## GPU (NVIDIA)

By default OpenMM runs on the CPU or OpenCL. For an NVIDIA GPU install the CUDA build of OpenMM through an extra, matching your CUDA version (`nvidia-smi` shows it):

```bash
pip install "pyLoopSage[cuda12]"    # CUDA 12
pip install "pyLoopSage[cuda13]"    # CUDA 13
uv pip install "pyLoopSage[cuda12]" # the same with uv
```

Do not install the separate `OpenMM-cuda` package: it is outdated and forces OpenMM 8.1.

## Optional features

Some inputs need a package that is not installed by default, so that a plain install never fails on a missing compiler or channel:

| Extra | Package | Needed for |
|---|---|---|
| `bigwig` | `pyBigWig` | BigWig signal tracks (`bw_files` in the config file) |
| `hic` | `hic-straw` | `.hic` reference maps (`--hic`); builds from source, needs a C++ compiler and libcurl when no wheel exists |
| `topology` | `topoly` | linking numbers (topology mode) |
| `all` | all three | everything |

For example `pip install "pyLoopSage[bigwig,hic]"`. If you use a feature without its package, LoopSage stops with a message that names the extra to install.

## Troubleshooting

- **pip downgrades OpenMM, or the solver reports a conflict:** use version 2.0.1 or newer. Older releases required `OpenMM-cuda`, which pins OpenMM 8.1.
- **`numpy.core.multiarray failed to import` or `_ARRAY_API not found`:** LoopSage needs numpy 2 (`numpy>=2.0`). A package in the same environment that was built for numpy 1 causes this message. Use a fresh environment.
- **Other numpy error:** a package was built against another numpy. Use a fresh environment and install everything in one command.

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

A `Dockerfile` with CUDA 12 support (installs `pyLoopSage[cuda12,all]`) is in the repository root:

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
