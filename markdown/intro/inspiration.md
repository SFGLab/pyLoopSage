# Inspiration

LoopSage is the first model I built and published on my own, as a first-year PhD student. I wanted chromatin modelling to be easier and friendlier to use, and I hoped to understand it from the first principles of biology.

## First principles

A first principle is a basic law that holds everywhere and needs no data from the system you study. In physics, a few such laws are enough to predict a lot. Take an ideal gas in a box: knowing only its temperature, volume and number of particles, the laws of physics give its pressure. It does not matter where the box is or what happened to the gas before.

When I started, I hoped chromatin could be treated in the same way: write down the basic rules of how cohesin and CTCF behave, and let the 3D genome follow from them.

## Why biology does not work like this

In a cell, the basic rules are not enough. Consider two cells of the same person, a liver cell and a neuron. Both contain the same DNA, and both obey the same laws of physics. Yet their genomes fold differently, because they differ in the information they carry: which genes are active, which proteins are bound to the DNA, and where CTCF sits and in which direction it points.

This information is not given by any law. It is the result of a long history of evolution and development, and the only way to know it is to measure it. In this sense a biological system is information plus physics.

| | What it gives | Where it comes from |
|---|---|---|
| Physics | how a polymer moves, how cohesin extrudes a loop, how CTCF blocks it | equations; the same in every cell |
| Information | which loops and anchors exist, which regions are active or inactive | measurements such as Hi-C, ChIA-PET and ChIP-seq; different in every cell type |

## What follows for modelling

Since the information cannot be derived, a chromatin model cannot be purely first-principles. It has to be driven by data, and it describes one biological case at a time: a given region, in a given cell type, under given conditions. The physics tells the model how the system can move, and the data tell it what the system is.

LoopSage follows this view. Loop extrusion and polymer mechanics are its engine, and the information is its input: the loops, the CTCF orientations, the compartments and the tracks that you provide. LoopSage does not explain why a loop exists. It asks which 3D structures are consistent with the loops seen in the data, and how well the result agrees with experiment.

## The gaps LoopSage tries to fill

Working with loop extrusion models, and talking with other researchers who build and use them, I noticed two gaps in the field. LoopSage is a modest attempt to narrow them, not a complete solution.

The first gap is validation. Chromatin models are rarely tested against experiment in a common, transparent way. A model usually reproduces the data it was tuned on, and it is seldom clear where it fails. Every model is perfectly imperfect, but without validation it is hard to say how.

The second gap is documentation and benchmarking. At the first international 4D Nucleome Hackathon in Seattle, our team compared several chromatin models on the same genomic region. We found that many of them are thinly documented, and that technical limits (different inputs, formats and assumptions) make a fair comparison difficult ([Kubica et al., *PLOS Computational Biology*, 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013358)).

## How LoopSage answers

LoopSage tries to address both openly. Its validation is transparent: every figure states how the simulation compares with the experiment. A clean optimizer fits the parameters to your data, and the report lets a reader check the strengths and weaknesses of the model.

Running many settings is meant to be easy, so that models can be benchmarked and their parameters optimized to give better results. Much remains to be improved, and this is the direction I hope LoopSage keeps following.
