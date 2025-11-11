# From-LIF-to-QIF

The data and code corresponds to experiments done in [From LIF to QIF: Toward Differentiable Spiking Neurons for Scientific Machine Learning](https://arxiv.org/abs/2511.06614). 

## Data
All data used are either given in '.mat' format or generated in the code provided.

## Usage:
The QIF neuron imeplementation on different examples and data are in each file, where the main file used to run ends with 'py'. 

For example, 'parabola/regressionpy.py' is the main file to run the code and 'parabola/regression.py' is the file that will be called by the main file. 

The module 'spikegd' called in files comes from [spikegd](https://github.com/chklos/spikegd) used in the paper [Smooth Exact Gradient Descent Learning in Spiking Neural Networks](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.027301) by Christian Klos and Raoul-Martin Memmesheimer. The code in this current repository is also built on it.


