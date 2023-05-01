# Dynamical-processes-on-networks
Patterns are omnipresent in nature: from spots or stripes on animal fur to neural activity in the human cortex. Many biological systems which exhibit such behaviour have been characterised by reaction-diffusion models, a theory pioneered by Alan Turing. Such models can be extended onto a network structure: interpreted as a meta-population model where each node carries a varying concentration of an activator and an inhibitor. The interaction of substances is constrained by the network's topology and hence its ability to sustain temporal patterns.

This repository contains supporting code for an essay which examines when such behaviour is observable when the network comprises of directed edges as opposed to undirected edges.

functions.py documents custom functions defining the reaction-diffusion models, methods to simulate such models on a network structure, a function to generate synthetic small-world networks using the Newman-Watts algorithm and a function which numerically computes the dispersion relation of each system, defining its ability to sustain Turing patterns.

Brusselator_model.ipynb and test.ipynb contain provisional experiments on synthetic networks, showing the range of patterns obervsable on directed and undirected networks.

chesapeake.ipynb, euroroads.ipynb, faa.ipynb, rheseus_macaques.ipynb and taro.ipynb contain simulations on empirical directed networks whose data lie in the networks folder and were obtained from konect.cc

Finally, introndlib.py is a bonus Python script where I used the NDLib software to generate epidemic models on networks

