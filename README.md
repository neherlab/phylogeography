# Analysis of phylogeographic inference

This repository contains the files associated with the manuscript entitled ``Lost in the woods: shifting habitats can lead phylogeography astray'', along with additional notes, and simulations.

## Simulations and figures

The simulations were run on the sciCORE cluster of the University of Basel and orchestrated using the Snakemake.
The figures in the manuscript were generated using the following scripts:

- Fig 1 -- illustration of spatial location of tips in case of free diffusion: [src/illustration.py](src/illustration.py)
- Fig 2 -- Sample size dependence of diffusion and velocity estimators: [src/sample_size_dependence.py](src/sample_size_dependence.py)
- Fig 3 -- Population heterogeneity and density regulation: [src/stable_density_figure.py](src/stable_density_figure.py). The input data for this script were precomputed using [src/stable_density.py](src/stable_density.py) as called by the workflow defined in the [Snakefile][Snakefile].
- Fig 4 -- Changing environments: Panels A-C are produced by [src/seasaw_example.py](src/seasaw_example.py), panels D and E by [src/plot_seasaw.py](src/plot_seasaw.py)
- Fig 5 -- Moving Oasis: Panels B-C are produced by [src/plot_waves.py](src/plot_waves.py).



