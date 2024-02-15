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
git clone https://gitlab.quandela.dev/swein/zero-photon-generator.git
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

To learn how to build and simulate photonic experiments, there are a set of tutorials available in the [Documentation](https://quandela.github.io/ZPGenerator).

# Backend
ZPGenerator was developed to circumvent a problem that arises when computing integrated detection probabilities for pulses of light, such as required to simulate the Hong-Ou-Mandel visibility of a quantum emitter, or source. The common approach (see below) results in an exponential scaling with the total number of photons detected by all photon-number resolving detectors, or by the number of threshold detectors, in the system. Moreover, since many time steps are often required to simulate accurate source dynamics (at least 100), the coefficient for this exponential scaling often prohibits the simulation of experiments involving more than two or three detectors. Such an approach cannot be effectively applied to larger photonic devices such as Quandela's Ascella QPU, which has 12 detectors.

## Solution
The name ZPGenerator represents the exploitation of a mathematical object called a Zero-Photon Generator (ZPG), which composes everything needed to solve a photo-counting experiment. The ZPG is analogous to the Hamiltonian of a closed quantum system and is closely related to the Lindbladian of an open-quantum system. The method is inspired by the quantum trajectories formalism, where source emission events and other interactions with the environment are treated stochastically. However, a key difference is that the ZPG is not solved using Monte Carlo methods, and fundamentally cannot simulate pure-state trajectories. Instead, a ZPG simulates something that can be referred to as a 'quantum corridor', which is a mixed-state trajectory or a collection of pure-state trajectories of the source state that evolve in time following a common property. In this case, the common property is the null result of our photo-counting experiment (i.e. every detector fails to detect at least one photon). Similar to quantum trajectories, the quantum corridor is not a normalized state. Rather, the trace of the quantum corridor gives the probability that the state passed through that corridor. Thus, the ZPG can be used to simulate the null result probability by integrating over a single temporal degree of freedom.

At first, it may seem like simulating only the null result is not enough to obtain all possible outcomes. However, the second key ingredient is that the ZPG, and hence the null probability, depends explicitly on the independent efficiencies of all the detectors. By choosing to evaluate the ZPG for a large enough set of unique detector efficiency configurations, it is possible to fully reconstruct the photon-number resolved detection probabilities \[[A. R. Rossi, S. Olivares, and M. G. A. Paris](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.055801)\]. 

How the ZPG is evaluated and how the reconstruct is accomplished can vary depending on the application. Interestingly, the solution to the ZPG produces a generating function G that is the Z-transform of the detection probability distribution P. The simple expression Z(P)=G thus summarizes the equation solved by ZPGenerator, giving a second meaning to its name. The result of all this is that ZPGenerator scales polynomially in the number of outcomes for a constant source Hilbert space. A detailed explanation of this method is available in the paper \[[S. C. Wein, Phys. Rev. A 109, 023713 (2024)](https://link.aps.org/doi/10.1103/PhysRevA.109.023713)\].


## The common approach
The common approach to the time-dynamic photon-counting problem is to first compute a multi-time correlation function of the emitted field amplitude using the _Quantum Regression Theorem_ (QRT) \[[A. Kiraz, M. Atatüre, and A. Imamoğlu](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.69.032305)\] along with an input-output relation \[[C. W. Gardiner and M. J. Collett](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.31.3761)\] to build a source-field operator correspondence. The correlation function approach holds in the lossy regime where field amplitudes can effectively be used to estimate detection probabilities. However, as efficiency improves, multi-photon components will survive to the detector, and the detector will no longer respond linearly to the signal intensity. Thus, it becomes necessary to explicitly evaluate detection probabilities, which can be done following a time-independent perturbation of the source dynamics by detector-induced jumps \[[H. Carmichael](https://books.google.fr/books?hl=en&lr=&id=uor_CAAAQBAJ&oi=fnd&pg=PR1&dq=open+systems+approach+carmichael&ots=O1fuOp1e0T&sig=90OrETtiChu91YRkK31SRZhDx1Q&redir_esc=y#v=onepage&q=open%20systems%20approach%20carmichael&f=false)\]. In either case, for pulsed experiments where detection probabilities (clicks) are generally not time-resolved, the temporal degrees of freedom that are necessary to describe the quantum behaviour of the light must be eventually traced out. Since each detector present in the experiment integrates over the arrival time of the pulse it is monitoring, this leads to an exponential scaling in the number of detectors or the number of photons detected by a single number-resolving detector. 
