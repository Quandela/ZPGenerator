# ZPGenerator
ZPGenerator provides an object-oriented interface and robust numerical backend for simulating quantum 
optical experiments using sources of pulsed quantum light that evolve in time following the laws of quantum physics. 
It implements a method based on the paper [[S. C. Wein, Phys. Rev. A 109, 023713 (2024)](https://link.aps.org/doi/10.1103/PhysRevA.109.023713)] to
efficiently simulate time-integrated photon detection of light produced by time-evolving emitters. These types of quantum-dynamic 
simulations demand a detailed understanding of the underlying physics and, using standard methods, often require a lot 
of time to code from scratch and are also computationally expensive to run. ZPGenerator aims to make these physics 
simulations more accessible and fast without the hefty knowledge overhead, allowing for quick prototyping of photonic 
experiments.
1) **Photonic Circuits** - quantum enthusiast, simulate photonic experiments using a catalogue components with pre-defined physics.
2) **Pulsed Sources** - quantum engineer, characterise and optimise catalogue source components by modifying their operation parameters.
3) **Custom Components** - quantum physics researcher, build custom components by combining catalogue components together.
4) **Custom Physics** - quantum dynamics expert, construct custom components from scratch, defined by a quantum master equation.

# Key Features

* Friendly python API to a state-of-the-art time-integrated photon counting physics simulator
* Fully composable and time dynamic quantum cascaded components
* Intuitive handling of quantum systems with many parameters
* Suite of quantum emitter characterisation tools
* Shaped laser pulses with quantum fluctuations
* States and channels conditioned on photonic measurements
* Modular architecture to add new catalogue components and features

# Installation

ZPGenerator requires:

* Above Python 3.8 and below Python 3.12

## GitLab
```bash
git clone https://github.com/Quandela/ZPGenerator.git
```
then to install ZPGenerator:
```bash
pip install .
```
Or for developers:
```bash
pip install -e .
```

## Package Use
Most simulations can be accomplished using just five classes of the repository.
- **Pulse**: creates and combines time-evolving functions used to control the source in order to emit light.
- **Source**: creates sources of light with characterisation methods to compute figures of merit such as quantum efficiency and single-photon purity.
- **Circuit**: creates basic linear-optical circuit components, such as a beamsplitter.
- **Detector**: creates detectors with a specified resolution, detection gate, and bin label.
- **Processor**: simulates one or more sources, whose emission may interfere in a linear-optical circuit, and evaluates detection correlations, conditional states, or quantum channels.

# Documentation

To learn how to build and simulate photonic experiments, there are a set of tutorials available in the [Documentation](https://quandela.github.io/ZPGenerator).
