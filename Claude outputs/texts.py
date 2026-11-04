"""Section introductions and glossary for the report."""

SECTION_INTRO = {
    "input": "What went into the model: the experimental loops or CTCF peaks, converted to bead resolution "
             "(one <em>simulation bead</em> = a fixed stretch of DNA, its size in bp is given in the tables).",
    "monte_carlo": "A stochastic sampler moves loop extrusion factors (LEFs) along the chromatin fibre and "
                   "accepts moves that lower an energy (Metropolis rule). The result is an ensemble of loop "
                   "configurations consistent with the data. Energies are in arbitrary units (a.u.); "
                   "positions are in beads and converted to bp/kb/Mb.",
    "molecular_dynamics": "Each loop configuration becomes a 3D polymer: beads are connected in a chain, loops "
                          "pull their two anchors together, and OpenMM integrates Langevin dynamics. Lengths "
                          "are in nm, energies in kJ/mol, time in fs/ps/ns.",
    "validation": "How well the simulated contacts reproduce the experimental loop signal. Correlations are "
                  "computed after removing positions where both signals are zero.",
    "knots": "Topology of the 3D structures: whether loops are linked with each other.",
}

GLOSSARY = [
    ("Bead", "One coarse-grained segment of the chromatin fibre. Its genomic size (bp per bead) is region length / number of beads."),
    ("LEF", "Loop extrusion factor (cohesin or condensin). It holds two legs (m, n) and the chromatin between them forms a loop."),
    ("Anchor", "A bead where a loop starts or ends, typically a CTCF binding site. L / R mark left / right anchors."),
    ("Monte Carlo (MC) step", "One sweep of 2 x (number of LEFs) random proposals. Snapshots are saved every <code>sample-every</code> steps."),
    ("Burn-in", "Initial MC steps discarded because the system has not yet equilibrated."),
    ("Metropolis / Annealing", "Metropolis keeps a constant temperature; Annealing lowers it from T<sub>init</sub> to T<sub>final</sub> to find low-energy states."),
    ("a.u.", "Arbitrary units: the MC energy and temperature have no physical unit; only ratios matter."),
    ("Contact map", "Matrix of how often two beads touch, comparable to a Hi-C map."),
    ("P(s)", "Mean contact probability at genomic separation s; real chromatin shows a steady power-law-like decay."),
    ("R<sub>g</sub>", "Radius of gyration: size of the 3D structure (nm)."),
    ("Pearson r / Spearman &rho;", "Correlation coefficients (&minus;1 to 1) for linear and rank agreement."),
    ("O/E", "Observed / expected: a value divided by the typical value at the same distance."),
    ("Linking number", "Integer counting how often two closed loops wind around each other (0 = unlinked)."),
]
