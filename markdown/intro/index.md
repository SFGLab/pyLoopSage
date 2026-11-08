![LoopSage](../assets/logo.svg){ .home-banner }

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

# What is LoopSage?

**LoopSage** is a program that turns a list of chromatin loops into an ensemble of 3D genome structures.
It models how the protein complex **cohesin** pushes DNA through itself and builds **loops**, and it checks the result against your data.

## The idea in one minute

Inside the cell nucleus, DNA is folded. A key part of the folding is **loop extrusion**: a ring-shaped protein complex (cohesin) lands on the chromatin fibre and slides along it, pulling in DNA and growing a loop. It stops at roadblocks, mainly **CTCF** proteins bound to specific DNA sites, which are called **anchors** of the loop.

LoopSage reproduces this process in a computer:

1. **Input.** You give the loops measured in an experiment (for example ChIA-PET, HiChIP or Hi-C loops) and a genomic region.
2. **Loop extrusion model.** Each cohesin is a *loop-extruding factor* (**LEF**), a pair of legs on the chromatin fibre. A **Monte Carlo** algorithm moves the legs at random and keeps moves that make the loops agree better with your data. After many steps it has produced many plausible loop configurations, an *ensemble*.
3. **3D structures.** A short **molecular dynamics** simulation turns every loop configuration into a 3D polymer structure.
4. **Checks.** The simulated contacts are compared with the experiment (loop strength, contact decay with distance, insulation score, compartments).
5. **Report.** Everything is collected in one HTML report with figures, tables and plain-language explanations.

## Words used in these notes

| Term | Meaning |
|---|---|
| **Bead** | One piece of the chromatin fibre in the model, covering a fixed stretch of DNA (for example 2 kb). The region is a chain of beads. |
| **Loop** | A pair of beads (left leg, right leg) held together by a LEF. |
| **Anchor** | A bead where loops tend to start or stop, typically a CTCF site. |
| **LEF** | Loop-extruding factor, the model of one cohesin complex. |
| **Monte Carlo (MC)** | A method that explores many configurations by random moves, accepting or rejecting each move with a probability set by an energy. |
| **Energy** | A number that is low when the loops agree with the data and with the physics of the model. Low-energy configurations are more likely. |
| **MD** | Molecular dynamics, a physical simulation of the 3D polymer. |
| **PET** | Paired-end tag, one sequencing read pair that supports a loop. More PET counts means a more reliable loop. |
| **BEDPE** | A text file format with one loop per line: two genomic intervals (the two anchors) plus a score. |

## What you give and what you get

**Input**

- A loop file: `.bedpe` (loops), or `.bed` / `.narrowPeak` (CTCF sites).
- A region, for example `chr1:77000000-95000000`.
- Optional: CTCF orientation (inside the BEDPE file, or read from a reference genome by LoopSage), BigWig tracks, a compartment track.

**Output** (in one folder per run)

- `report/report.html`, the full report with figures and explanations.
- `visualizations/`, every figure as PNG and SVG.
- `structures/`, 3D structures (`.cif`), an interactive 3D viewer and videos.
- `data/`, the numerical results; maps are saved as `.npy` files in `data/metadata/`.

## Quick start

```bash
loopsage -f loops.bedpe -r chr1:77000000-95000000
```

Then open `results/chr1_77000000_95000000_1/report/report.html` in a browser.
Install the program first: see [Installation](installation.md). To learn why LoopSage is built the way it is, read the [Inspiration](inspiration.md). All options are listed in the [Reference](../reference/index.md).

## Citation

If you use LoopSage, please cite:

- Korsak, S. and Plewczynski, D. *LoopSage: An energy-based Monte Carlo approach for the loop extrusion modelling of chromatin.* Methods (2024).
- Jodkowska, K., Parteka-Tojek, Z., Agarwal, A., Denkiewicz, M., Korsak, S., Chiliński, M., Banecki, K., and Plewczynski, D. *Improved cohesin HiChIP protocol and bioinformatic analysis for robust detection of chromatin loops and stripes.* bioRxiv (2024). [doi:10.1101/2024.05.16.594268](https://doi.org/10.1101/2024.05.16.594268)
