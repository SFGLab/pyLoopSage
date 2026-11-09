# Theory

Welcome! This part of the documentation explains the ideas behind LoopSage, from the biology of loop extrusion to the physics and statistics that turn it into a working model. You do not need a particular background to read it. Every new word and every symbol is explained when it first appears, so a biologist can follow the mathematics, and a physicist or mathematician can follow the biology.

The pages build on each other, but each one can also be read alone.

- [Scope](scope.md): what we want the model to explain, and how a handful of simple rules can produce the patterns of a Hi-C contact map.
- [Loop extrusion](loop-extrusion.md): the molecular players (cohesin, CTCF, condensin), the loop extrusion model and its assumptions, TADs, loop dots and stripes, and why a snapshot is not enough and dynamics is needed.
- [Mechanics](mechanics.md): Hamiltonian, phase space, ensembles and the Boltzmann law, and why an energy-based model is a reasonable approximation for a living cell that is not in equilibrium.
- [MCMC](mcmc.md): Markov chains, detailed balance, ergodicity, ergodic means, burn-in and autocorrelation: how the sampler works and how to tell that it worked.
- [Molecular dynamics](molecular-dynamics.md): the polymer, the force field, Newton's second law and the Langevin equation: how a set of loops becomes a 3D structure.
- [Topology](topology.md): knots, links and the linking number, how it is computed in 3D, and how LoopSage checks whether loops are interlocked (optional and advanced).

!!! tip "Where to start"
    If you are new to the field, read the pages in the order above. If you only want to know what a setting does, go to the [reference](../reference/index.md) or the [implementation](../implementation/index.md) pages, which link back here whenever an idea needs more explanation.
