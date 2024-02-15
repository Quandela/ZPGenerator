Welcome to the ZPGenerator documentation!
=========================================

The ZPGenerator package provides an object-oriented interface and robust numerical backend for simulating quantum
optical experiments using sources of pulsed quantum light that evolve in time following the laws of quantum physics.

It implements a method that exploits zero-photon measurements to efficiently simulate photon detection of light
produced by time-evolving emitters. These types of quantum-dynamic simulations demand a detailed understanding of the
underlying physics and, using standard methods, often require a lot of time to code from scratch and are also
computationally expensive to run. ZPGenerator aims to make these physics simulations accessible and fast without the
hefty knowledge overhead, allowing for quick prototyping of photonic experiments.

Currently, ZPGenerator uses `QuTiP`_ to manipulate quantum states and operators.
Please refer to the `QuTiP documentation`_ whenever necessary.

If you are using ZPGenerator for academic work, please cite the `paper introducing the backend method`_ as:

.. _QuTiP: https://qutip.org/
.. _QuTiP documentation: https://qutip.org/docs/latest/
.. _paper introducing the backend method: https://link.aps.org/doi/10.1103/PhysRevA.109.023713

.. code:: latex

   @article{wein2024simulating,
     title={Simulating photon counting from dynamic quantum emitters by exploiting zero-photon measurements},
     author={Wein, Stephen C},
     journal={Physical Review A},
     volume={109},
     pages={023713},
     year={2024},
     publisher={APS}
   }

.. toctree::
   :caption: Documentation
   :maxdepth: 2
   :hidden:

   installation
   notebooks/components
   notebooks/parameters
   notebooks/pulses
   notebooks/processors
   notebooks/backend

.. toctree::
   :caption: Catalogue Components
   :maxdepth: 2
   :hidden:

   notebooks/sources_catalogue
   notebooks/circuits_catalogue
   notebooks/detectors_catalogue

.. toctree::
   :caption: Basic Tutorials
   :maxdepth: 2
   :hidden:

   notebooks/photonic_circuits
   notebooks/pulsed_sources
   notebooks/quantum_dots
   notebooks/cavity_QED

.. toctree::
   :caption: Advanced Tutorials
   :maxdepth: 2
   :hidden:

   notebooks/fibonacci_states
   notebooks/wigner_functions
   notebooks/entanglement_generation
   notebooks/RUS_gate
   notebooks/component_construction

.. toctree::
   :caption: Code Reference
   :maxdepth: 2
   :hidden:

   reference/code_reference
