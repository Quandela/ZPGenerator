from zpgenerator.components import Detector, Source
from zpgenerator.simulate import Processor
from math import isclose


def test_bin_array_init():
    array = Detector.partition(thresholds=[0, 1, 2])
    p = Processor() // Source.fock(1) // array
    pn = p.probs()
    expected_pn = {(0, 0, 1): 0.13533459399621803, (0, 1, 0): 0.23254539677812022, (1, 0, 0): 0.6321207415902489}
    assert all(isclose(pn[k], expected_pn[k], abs_tol=1e-4) for k in expected_pn.keys())

    array = Detector.partition(thresholds=['t0', 't1', 't2'], parameters={'t0': 0, 't1': 1, 't2': 2})
    p = Processor() // Source.fock(1) // array
    pn = p.probs()
    expected_pn = {(0, 0, 1): 0.13533459399621803, (0, 1, 0): 0.23254539677812022, (1, 0, 0): 0.6321207415902489}
    assert all(isclose(pn[k], expected_pn[k], abs_tol=1e-4) for k in expected_pn.keys())

    expected_pn = {(0, 0, 1): 0.36787926424147155, (0, 1, 0): 0.23865169828662577, (1, 0, 0): 0.3934696212744341}
    pn = p.probs(parameters={'t0': 0, 't1': 0.5, 't2': 1})
    assert all(isclose(pn[k], expected_pn[k], abs_tol=1e-4) for k in expected_pn.keys())

    array = Detector.partition(thresholds=[[0, 'tf'], [5, 10]], parameters={'tf': 1})
    p = Processor() // Source.fock(1) // array
    pn = p.probs()
    expected_pn = {(0, 0): 0.361187, (0, 1): 0.006693, (1, 0): 0.63212}
    assert all(isclose(pn[k], expected_pn[k], abs_tol=1e-4) for k in expected_pn.keys())

    pn = p.probs(parameters={'tf': 2})
    expected_pn = {(0, 0): 0.12864387135612868, (0, 1): 0.006692994168018513, (1, 0): 0.8646632516130764}
    assert all(isclose(pn[k], expected_pn[k], abs_tol=1e-4) for k in expected_pn.keys())


def test_bin_labels():
    p = Processor() // Source.fock(1)
    p.add(0, Detector.pnr(2), bin_name='bin')
    assert p.bin_labels == ['bin']

