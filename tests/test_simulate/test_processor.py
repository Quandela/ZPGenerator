from zpgenerator.simulate.processor import Processor
from zpgenerator.elements import *
from zpgenerator.network import DetectorGate
from numpy import exp, log
from math import isclose
from qutip import Qobj


def make_emitter_splitter():
    emitter = TwoLevelEmitter(name='TLS')

    p = Processor()
    p.add(0, emitter)

    assert p.modes == 1
    assert p.component.subdims == [2]

    p.add(0, BeamSplitter(name='BS'))

    assert p.modes == 2

    p.add(0, DetectorGate(resolution=2), bin_name='D')

    p.precision = 12
    p.initial_state = emitter.states['|e>']
    p.final_time = 3
    p.initial_time = 0
    return p


def assert_prbs(processor: Processor, expected_prbs, time=None):
    if time is not None:
        processor.final_time = time
    prbs = processor.probs(chop=True)
    expected_prbs = expected_prbs(processor.final_time if time is None else time)
    assert all(isclose(prbs[k], expected_prbs[k], abs_tol=10 ** -(processor.precision)) for k in expected_prbs.keys())


def test_processor_base_simple_decay_probs():
    p = make_emitter_splitter()

    assert p.component.subdims == [2]
    assert p.modes == 2
    assert p.bins == 1
    assert p.bin_labels == ['D']

    expected = lambda t: {0: 1 - (1 - exp(-t)) / 2, 1: (1 - exp(-t)) / 2}

    assert_prbs(p, expected, 1)  # testing propagation to time = 1
    assert_prbs(p, expected, 2)  # testing propagation from t=1 to t=2
    assert_prbs(p, expected, 1)  # testing propagation from t=0 to t=2 using automatic reset
    assert_prbs(p, expected, 4)  # testing propagation beyond final_time (manually specified)


def test_processor_base_simple_decay_binned():
    p = make_emitter_splitter()
    p.add(1, BeamSplitter())
    p.add(1, DetectorGate(resolution=2), bin_name='D')

    assert p.final_time == 3

    assert p.bins == 1
    assert p.bin_labels == ['D']

    expected = lambda t: {0: 1 - 3 * (1 - exp(-t)) / 4, 1: 3 * (1 - exp(-t)) / 4}
    assert_prbs(p, expected)  # testing if outputs from both detectors are added together


def test_processor_base_simple_decay_correlations():
    p = make_emitter_splitter()
    p.add(1, BeamSplitter())
    p.add(1, DetectorGate(resolution=1), bin_name='D2')

    assert p.bins == 2
    assert p.bin_labels == ['D', 'D2']

    expected = lambda t: {(0, 0): 1 - 3 * (1 - exp(-t)) / 4,
                          (1, 0): (1 - exp(-t)) / 2,
                          (0, 1): (1 - exp(-t)) / 4}

    assert_prbs(p, expected)


def test_processor_base_simple_decay_states():
    p = make_emitter_splitter()
    p.final_time = 1

    states = p.conditional_states(chop=True)

    assert isclose((states[0] - Qobj([[(1 - exp(-1)) / 2, 0], [0, exp(-1)]])).tr(), 0, abs_tol=1e-6)
    assert isclose((states[1] - Qobj([[(1 - exp(-1)) / 2, 0], [0, 0]])).tr(), 0, abs_tol=1e-6)


def test_processor_base_simple_decay_channels():
    p = make_emitter_splitter()
    p.final_time = 1

    channels = p.conditional_channels(basis=[p.component._elements[0].states['|g>'],
                                             p.component._elements[0].states['|e>']])

    ch0 = Qobj([[1, 0, 0, (1 - exp(-1)) / 2],
                [0, exp(-1 / 2), 0, 0],
                [0, 0, exp(-1 / 2), 0],
                [0, 0, 0, exp(-1)]])

    assert all(isclose(abs(channels[0][i, j]), abs(ch0[i, j]), abs_tol=10 ** -p.precision)
               for i in range(4) for j in range(4))

    ch1 = Qobj([[0, 0, 0, (1 - exp(-1)) / 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

    assert all(isclose(abs(channels[1][i, j]), abs(ch1[i, j]), abs_tol=10 ** -p.precision)
               for i in range(4) for j in range(4))


def test_unnormalised_initial_state():
    emitter = TwoLevelEmitter(name='TLS')

    p = Processor() // emitter // DetectorGate(1, gate=[0, log(2)])

    p.precision = 12
    p.initial_state = Qobj([[0, 0], [0, 0.5]])
    p.initial_time = 0

    prbs = p.probs()
    expected_prbs = {(0,): 0.25, (1,): 0.25}
    assert all(isclose(prbs[k], expected_prbs[k], abs_tol=10 ** -(p.precision)) for k in expected_prbs.keys())


def test_sequential_conditional_states():
    emitter = TwoLevelEmitter(name='TLS')

    p = Processor()
    p.add(0, emitter)
    p.add(0, DetectorGate(1, gate=[0, log(2)]))

    p.precision = 12
    p.initial_state = emitter.states['|e>']
    p.initial_time = 0

    prbs = p.probs()
    assert isclose(prbs[1], 1/2, abs_tol=10 ** -(p.precision))

    states = p.conditional_states()
    p.initial_state = states[1]
    prbs = p.probs()
    isclose(prbs[0], 1 / 2, abs_tol=10 ** -(p.precision))

    p.initial_state = states[0]
    prbs = p.probs()
    isclose(prbs[0], 1 / 4, abs_tol=10 ** -(p.precision))