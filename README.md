# Neuronal selectivity for multiple features in the primary visual cortex
Wenqing Wei, Benjamin Merkt, Stefan Rotter

This is the code used for running simulations and visualizing results in the [paper](https://doi.org/10.1101/2022.07.18.500396).

## Getting Started
The Python files within this repository whose names commence with ‘fig_’ comprise three parts:
1. Numerical (NEST) / Analytical simulation
2. Analysis of the simulation data
3. Visualisation, i.e. plotting out the figures in the paper

### Prerequisites
Python 2.7 or later (Packages: numpy, scipy, matplotlib, etc)\
[NEST 2.20.0](https://www.nest-simulator.org)

### Resources
The core part of the code has been tested on a local PC (4 Core with 8 GB of Memory). It takes up to few minutes for running the numerical simulation of the entire feedforward network once,
i.e. for one stimulus orientation. For generating transfer function of a single LIF neuron, it is also quite fast with the local PC. 
However, for analysing the effect of different parameters on the neuronal responses, investigating the input-output transformation of the network, extracting receptive fields of neurons, much more computational resources are needed. In this case, running simulaitons for different stimulus angles, seeds and parameters in parallel can help to save a lot of time.
To produce the original results, I have used parallelization in High Performance Computing environment.

## How To Use
To clone and run this project, you'll need Git installed on your computer. From your command line:
```python
# Clone this repository
$ git clone https://github.com/whisper186/Neuronal_Selectivity_for_Multiple_Features.git

# Go into the repository
$ cd Simulation-NeuronalSelectivityforMultipleFeatures
```
Then run the respective Python file (name starting with 'fig_') of producing the figure that you are interested in.

## Contact
Wenqing Wei - wenqing_wei@outlook.com\
Stefan Rotter - stefan.rotter@bio.uni-freiburg.de



