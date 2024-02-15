# from zpgenerator import Pulse, Source, Material
# from numpy import pi
# from math import isclose
#
# def test_phonon_assisted_excitation():
#     source = Source.phonon_assisted(pulse=Pulse.gaussian({'width': 10, 'area': 20 * pi}),
#                                     parameters={'resonance': -0.5,  # we can shift the rotating frame
#                                                 'dephasing': 0.01},  # pure dephasing
#                                     purcell_factor=10,
#                                     regime=0.1,  # the bad coupling regime parameter (lower == more bad)
#                                     timescale=200,  # The Purcell-enhanced decay timescale in ps
#                                     temperature=7,  # The temperature is in Kelvin
#                                     material=Material.ingaas_quantum_dot())
#
#     assert isclose(source.mu(), 0.765, abs_tol=1e-2)
#     assert isclose(source.mu(parameters={'resonance': 0}), 0.466, abs_tol=1e-2)
#     assert isclose(source.mu(parameters={'resonance': 0, 'detuning': 0.5}), 0.765, abs_tol=1e-2)
#
#     assert isclose(source.g2(), 0.0294, abs_tol=1e-2)
#     assert isclose(source.g2(parameters={'resonance': 0}), 0.260, abs_tol=1e-2)
#     assert isclose(source.g2(parameters={'resonance': 0, 'detuning': 0.5}), 0.0294, abs_tol=1e-2)
#
#
# from zpg import *
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# source = Source.phonon_assisted(pulse=Pulse.gaussian({'area': 20 * np.pi, 'width': 10}),
#                                 parameters={'resonance': -0.5,  # we can shift the rotating frame
#                                             'dephasing': 1/3300},  # pure dephasing
#                                 purcell_factor=10,
#                                 regime=0.1,  # the bad coupling regime parameter (lower == more bad)
#                                 timescale=200,  # The Purcell-enhanced decay timescale in ps
#                                 temperature=7,  # The temperature is in Kelvin
#                                 material=Material.ingaas_quantum_dot())
#
# detunings = np.linspace(-10, 10, 20)
# mu_set0 = [source.mu(parameters={'resonance': -x}) for x in detunings]
# mu_set1 = [source.mu(parameters={'detuning': x, '_resonance': 0}) for x in detunings]