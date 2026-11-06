# Implementation

How LoopSage works inside: the flow of the program and, for each stage, what it does, which parameters control it and where the code is. The pages follow the order in which the stages run.

- [The Pipeline](pipeline.md): the flow chart of LoopSage, the stages, how they hand their results to each other, how to run one stage alone and what the run folder contains.
- [Preprocessing](preprocessing.md): how BEDPE loops, CTCF peaks, BigWig tracks and Hi-C maps are read and modelled, and the two binding modes (preferential and contrastive).
- [The Motif Finder](motif-finder.md): how the orientation of CTCF sites is read from the DNA sequence of the loop anchors.
- [The Optimizer](optimizer.md): the mathematics of the parameter search (loss, Nelder&ndash;Mead, early stopping) and how to use it.
- [Monte Carlo](monte-carlo.md): the energy of LoopSage, the moves (rebinding, sliding, epigenetic changes) and all parameters of the sampler.
- [Molecular Dynamics](molecular-dynamics.md): how a loop snapshot becomes a 3D structure: forces, the Langevin integrator and the simulation arguments.
