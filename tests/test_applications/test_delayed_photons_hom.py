from zpgenerator import *
from math import isclose


def test_component_delays():
    qpu = Processor()
    qpu.add(0, Source.fock(1, name='mode 0'))
    qpu.add(1, Source.fock(1, name='mode 1'))
    qpu.add(0, Circuit.bs())
    qpu.add([0, 1], Detector.threshold())

    p11_pos = qpu.probs(parameters={'mode 1/delay': 0.5})[1, 1]
    p11_neg = qpu.probs(parameters={'mode 1/delay': -0.5})[1, 1]
    p11_zero = qpu.probs(parameters={'mode 1/delay': 0})[1, 1]
    p11_dist = qpu.probs(parameters={'mode 1/delay': 0, 'dephasing': 100})[1, 1]

    assert isclose(p11_pos, 0.196735, abs_tol=1e-4)
    assert isclose(p11_neg, 0.196735, abs_tol=1e-4)
    assert isclose(p11_zero, 0, abs_tol=1e-4)
    assert isclose(p11_dist, float(1 / 2 - 1 / (1 + 200)), abs_tol=1e-4)
