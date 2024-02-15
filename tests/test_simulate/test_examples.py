from math import isclose
from zpgenerator.simulate import *
from zpgenerator.components import *
from qutip import Qobj

def assert_probs(qpu, expected: dict):
    for k, v in expected.items():
        assert isclose(qpu.probs()[k], v, abs_tol=1e-3)


def test_perfect_hom_pnr():
    p = Processor()
    p.add([0, 1], Source.fock(1))
    p.add(0, Circuit.bs())
    p.add(0, Detector.pnr(2, bin_name='L'))
    p.add(1, Detector.pnr(2, bin_name='R'))

    assert p.bin_labels == ['L', 'R']
    assert_probs(p, {(2, 0): 1 / 2, (0, 2): 1 / 2})


def test_processor_hom_threshold():
    p = Processor()
    p.add([0, 1], Source.fock(1))
    p.add(0, Circuit.bs())
    p.add(0, Detector.threshold(bin_name='L'))
    p.add(1, Detector.threshold(bin_name='R'))

    assert p.bin_labels == ['L', 'R']
    assert_probs(p, {(1, 0): 1 / 2, (0, 1): 1 / 2})


def test_processor_hom_mixed_detectors():
    p = Processor()
    p.add([0, 1], Source.fock(1))
    p.add(0, Circuit.bs())
    p.add(0, Detector.pnr(2, bin_name='PNRD'))
    p.add(1, Detector.threshold(bin_name='TD'))

    assert p.bin_labels == ['PNRD', 'TD']
    assert_probs(p, {(2, 0): 1 / 2, (0, 1): 1 / 2})