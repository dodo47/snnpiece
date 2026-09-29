# SNNPiece: Analysing spiking neural networks using causal pieces

Code accompanying the publication ["Causal pieces: analysing and improving spiking neural networks piece by piece"](https://doi.org/10.48550/arXiv.2504.14015).

## Abstract

We introduce _causal pieces_, a novel concept for analysing spiking neural networks (SNNs), inspired by _linear pieces_ used to study expressivity and trainability in artificial neural networks (ANNs). Causal pieces partition the input and parameter space of a feedforward SNN with single-spike coding into distinct regions where the same subnetwork causes the output spikes. For networks of current-based leaky integrate-and-fire (LIF) neurons with large membrane time constants, we show that within each causal piece, output spike times are locally Lipschitz continuous with respect to inputs and network parameters. We further prove a lower bound on the approximation error that depends on the number of causal pieces. Thus, the number of causal pieces is a measure of the approximation capabilities of SNNs, which is valid despite spike-time discontinuities and applies to networks with both excitatory and inhibitory synapses. 
Empirically, we find that parameter initialisations yielding more causal pieces on the training set strongly correlate with SNN training success across multiple benchmarks, including Yin-Yang, Fashion-MNIST, and EuroSAT. Moreover, simulations with standard single-spike LIF neurons indicate that our findings extend beyond the theoretically analysed regime. These results establish causal pieces as a powerful and principled tool for analysing and improving the computational capabilities of SNNs.

## Code

We provide implementations of neuron models, routines for training spiking neural networks, and methods for evaluating causal pieces.

To use our code, install the provided package snnpiece via pip
```
pip install -e .
```
from the folder containing the `setup.py`. 
The source code is located in `snnpiece/`. 
Example notebooks can be found in `Examples/`. 

The following neuron models are supported. They all use single-spike coding. Spike times are calculated analytically.
- Simple spike-respone model with linear kernel ([Neuman et al. 2025](https://doi.org/10.48550/arXiv.2404.04549))
- Non-leaky integrate-and-fire neuron model ([Mostafa 2017](https://doi.org/10.48550/arXiv.1606.08165))
- Leaky integrate-and-fire neuron model for $\tau_m = 2 \tau_s$ ([Göltz et al. 2021](https://doi.org/10.48550/arXiv.1912.11443))

For training, real gradients are calculated from the analytic spike times using automatic differentiation.

Included datasets are: [Yin-Yang](https://github.com/lkriener/yin_yang_data_set), MNIST, [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist#benchmark), [EuroSAT](https://github.com/artemisart/EuroSAT-image-classification/tree/master)

## Citation

If you use the code, or find it helpful for your own work, please cite

```
@article{dold2026pieces,
  title={Causal pieces: analysing and improving spiking neural networks piece by piece},
  author={Dold, Dominik and Petersen, Philipp},
  journal={Accepted at NeurIPS 2026.},
  year={2026},
  doi={10.48550/arXiv.2504.14015},
}
```
