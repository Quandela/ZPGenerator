from zpgenerator.dynamic.operator.phonon_bath import *
from zpgenerator.dynamic.pulse import Pulse
from math import isclose
from qutip import destroy, fock
from numpy import sqrt


def test_material():
    material = Material.ingaas_quantum_dot(electron_confinement=4.2e-9, hole_confinement=4.2e-9)
    assert isclose(material.spectral_density(1), 0.0259257, abs_tol=1e-4)
    assert isclose(material.polaron_shift, 0.08201, abs_tol=1e-4)


def test_bath():
    pulse = Pulse.gaussian()

    bath = PhononBath(material=Material.ingaas_quantum_dot(electron_confinement=4.2e-9, hole_confinement=4.2e-9),
                      temperature=7,
                      resolution=200,
                      max_power=20)

    assert isclose(bath.polaron_shift, 0.08201, abs_tol=1e-4)
    assert isclose(bath.coth(2), 1.03731, abs_tol=1e-4)

    bath.initialize()

    assert isclose(bath._rs(1), -0.0322981, abs_tol=1e-4)
    assert isclose(bath._is(1), -0.020362, abs_tol=1e-4)
    assert isclose(bath._rc(1), 0.0409553, abs_tol=1e-4)
    assert isclose(bath._ic(1), -0.0537046, abs_tol=1e-4)
    assert isclose(bath.gamma_star(1, parameters={'resonance': 0}), sqrt(0.163545), abs_tol=1e-4)

    peak_power = sqrt(abs(pulse.evaluate(0))**2 + bath.polaron_shift**2)
    assert isclose(bath._rs(peak_power), 0.00862788, abs_tol=1e-4)
    assert isclose(bath._is(peak_power), 0, abs_tol=1e-4)
    assert isclose(bath._rc(peak_power), 0, abs_tol=1e-4)
    assert isclose(bath._ic(peak_power), 0.0012182, abs_tol=1e-4)
    assert isclose(bath.gamma_star(peak_power, parameters={'resonance': 0}), 0, abs_tol=1e-4)

    assert isclose(bath._phi0, 0.115401, abs_tol=1e-4)
    assert isclose(bath.attenuation, 0.943933, abs_tol=1e-4)


def test_environment():
    pulse = Pulse.gaussian()
    bath = PhononBath(material=Material.ingaas_quantum_dot(), temperature=7, max_power=25)
    env = bath.build_environment(pulse=pulse, transition=destroy(2))

    assert env.uses_parameter('resonance')

    assert isclose(abs((env.evaluate(0, {'resonance': -bath.polaron_shift})(fock(2)) * destroy(2)).tr()),
                   0.00862788, abs_tol=1e-4)

    assert isclose(real(pulse.evaluate(0, {'resonance': 1 - bath.polaron_shift})),
                   12.5331, abs_tol=1e-4)
    assert isclose(imag(env.evaluate(0, {'resonance': 10 - bath.polaron_shift})[1, 0]),
                   0.01514162226773711, abs_tol=1e-4)
    assert isclose(imag(env.evaluate(0, {'resonance': 10 - bath.polaron_shift})[1, 3]),
                   0.02554197973734049, abs_tol=1e-4)


    # assert emitter.evaluate(3) == []