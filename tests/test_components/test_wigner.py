from zpgenerator.components.detector import Detector
from zpgenerator.simulate import Processor
from zpgenerator.components import Source
from math import isclose
from numpy import real


def test_parity():
    p = Processor() // Source.fock(1) // Detector.parity()
    pn = p.probs()
    assert isclose(real(pn['p']), -1, abs_tol=1e-5)

    p = Processor() // Source.fock(2) // Detector.parity()
    pn = p.probs()
    assert isclose(real(pn['p']), 1, abs_tol=1e-5)
