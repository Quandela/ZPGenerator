from zpgenerator.virtual.generator import *
from zpgenerator.virtual.state import VState
from zpgenerator.virtual.configuration import PhysicalDetectorGate
from zpgenerator.virtual.tree import VTree
from zpgenerator.virtual.branch import MeasurementBranch
from zpgenerator.network.detector import TimeBin
from zpgenerator.elements import Emitter
from zpgenerator.time import TimeOperator, Operator, TimeIntervalFunction, TimeInterval
from qutip import destroy, create, fidelity, Qobj
from numpy import pi
from pytest import approx


def semiclassical_driving_hamiltonian(parameters: dict):
    return parameters['area'] / parameters['width'] * (destroy(2) + create(2)) / 2


def window(parameters: dict):
    return [parameters['delay'] - parameters['width'] / 2, parameters['delay'] + parameters['width'] / 2]


def test_generator_time_independent():
    emitter = Emitter.two_level(modes=2)

    interval = TimeInterval(interval=window, parameters={'delay': 0, 'width': 0.1})
    emitter.hamiltonian.add(TimeOperator(operator=Operator(semiclassical_driving_hamiltonian,
                                                           parameters={'area': pi, 'width': 0.1}),
                                         functions=TimeIntervalFunction(value=1, interval=interval)))

    gen = Generator(component=emitter)

    parameters = {'delay': 0.5, 'width': 0.2}

    times = emitter.times(parameters)
    assert times == [0.4, 0.6]

    istate = emitter.states['|g>']
    itime = times[0]
    vstate = VState(state=istate, time=itime)

    vprop = gen.build_propagator(itime, parameters)
    # assert isinstance(vprop, VPropTI)  # commented out since we use VPropHTD instead (much faster)
    assert vprop.jumps == []

    vstate.propagate(vprop, times[1])
    assert approx(fidelity(vstate, Qobj([[0.07153569 + 0.j, 0 + 0.05927914j],
                                         [0. - 0.05927914j, 0.92846431 + 0.j]]))) == 1

    vstate = VState(state=istate, time=itime)
    vdetector = PhysicalDetectorGate(resolution=5, efficiency=0.5, gate=[itime, 5])
    times = sorted(list(set(times + vdetector.gate.times(parameters))))
    assert times == [0.4, 0.6, 5]

    vtree = VTree(initial_state=vstate)
    vtree.add_branch(MeasurementBranch([TimeBin(vdetector, mode=0)]))
    vtree.add_branch(MeasurementBranch([TimeBin(vdetector, mode=1)]))

    # prop = factory.build_propagator(times[0], parameters)
    # assert prop.jumps == []
    # vtree.propagate(prop, times[1], parameters)
    #
    # vtree.propagate(times[2], factory.build_propagator(times[1], parameters), parameters)
    #
    # vtree.get_states()
    # vtree.get_points()
