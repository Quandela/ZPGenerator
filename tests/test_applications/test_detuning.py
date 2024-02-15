from zpgenerator import *
from math import isclose
from numpy import pi

def test_detuned_tls_gaussian_pulse():
    pulse = Pulse.gaussian()
    source = Source.two_level(pulse=pulse)

    assert isclose(source.mu(parameters={'resonance': 1}), source.mu(parameters={'detuning': 1}), abs_tol=1e-4)
    assert isclose(source.g2(parameters={'resonance': 1.5}), source.g2(parameters={'detuning': 1.5}), abs_tol=1e-4)


def test_detuned_exciton_gaussian_pulse():
    pulse = Pulse.gaussian()
    source = Source.exciton(pulse=pulse, parameters={'fss': 1})

    assert not isclose(source.mu(parameters={'resonance': 0.5}), source.mu(parameters={'detuning': 0.5}), abs_tol=1e-4)
    assert isclose(source.mu(parameters={'resonance': 0.5}), source.mu(parameters={'detuning': -0.5}), abs_tol=1e-4)


def test_detuned_tpe_gaussian_pulse():
    pulse = Pulse.gaussian()
    source = Source.biexciton(pulse=pulse, parameters={'binding': 100})

    assert isclose(source.mu(parameters={'resonance': -50, 'area': 4 * pi}),
                   source.mu(parameters={'detuning': 50, 'area': 4 * pi}), abs_tol=1e-4)
    assert isclose(source.mu(parameters={'resonance': 25, 'area': 5 * pi}),
                   source.mu(parameters={'detuning': -25, 'area': 5 * pi}), abs_tol=1e-4)
