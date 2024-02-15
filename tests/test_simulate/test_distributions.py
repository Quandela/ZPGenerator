from zpgenerator.simulate.algorithms.distributions import *
from qutip import fock


def test_correlation_distribution():
    pn = CorrelationDistribution({(1, ): 1})
    assert pn[1] == 1

    pn = CorrelationDistribution({(2, 0): 2})
    assert pn[2, 0] == 2
    assert pn[2, 0, 3] == 0


def test_state_distribution():
    sn = Distribution({(1, ): fock(1)})
    assert sn[1] == fock(1)


