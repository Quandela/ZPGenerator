from zpgenerator.system.multibody import *
from zpgenerator.system.natural import HamiltonianBase
from zpgenerator.time import unitary_propagation_superoperator
from test_control import _make_controlled_system
from test_natural import _make_natural_system
from test_coupling import _make_jaynes_cummings
from qutip import Qobj, fock, destroy, create, liouvillian, qzero, sprepost, qeye, num, tensor
from copy import deepcopy
from numpy import pi, sqrt
from zpgenerator.time.parameters import Parameters


d = Parameters.DELIMITER


def test_multibodysystem_init_empty():
    system = MultiBodyEmitterBase(name='pair')
    assert system.subdims is None
    assert system.dim is None
    assert system.states == {}
    assert system.operators == {}
    assert system.bodies == 0
    assert system.name == 'pair'
    assert system.evaluate(0) == Qobj()


def _make_system():
    sys0 = _make_natural_system()
    sys0.name = 'system 0'
    sys0.states = {'g': fock(2, 0), 'e': fock(2, 1)}
    sys0.operators = {'I': qeye(2), 'X': destroy(2) + create(2)}
    return sys0


def test_multibodysystem_init_onebody():
    sys0 = _make_system()
    system = MultiBodyEmitterBase(subsystems=sys0, name='pair')

    assert system.states == sys0.states
    assert system.operators == {'I': qeye(2), 'X': destroy(2) + create(2)}
    assert system.dim == 2
    assert system.subdims == [2]
    assert system.bodies == 1
    assert system.evaluate(0) == liouvillian(H=qzero(2), c_ops=[destroy(2)])
    assert system.partial_evaluate(0)[1].evaluate(0) == destroy(2)
    assert system.evaluate_quadruple(0).environment[0].constant == destroy(2)
    assert system.evaluate_dirac(0).evaluate() == sprepost(qeye(2), qeye(2))


def _make_twobody_system():
    sys0 = _make_system()
    sys1 = deepcopy(sys0)
    sys1.default_parameters = {'decay': 2, 'detuning': 1}
    sys1.name = 'system 1'
    system = MultiBodyEmitterBase(subsystems=sys0)
    assert system.subsystems['system 0'] == sys0

    system.add(sys1)
    return system


def _make_controlled_twobody_system():
    sys0 = _make_controlled_system()
    sys0.name = 'emitter'
    sys1 = _make_system()
    sys1.name = 'atom'
    return MultiBodyEmitterBase(subsystems=[sys0, sys1])


def test_system_collection_methods():
    system = _make_controlled_twobody_system()

    assert system.parameters == ['atom' + d + 'decay',
                                 'atom' + d + 'detuning',
                                 'emitter' + d + 'decay' + d + 'rate',
                                 'emitter' + d + 'dephasing' + d + 'rate',
                                 'emitter' + d + 'detuning',
                                 'emitter' + d + 'pulse' + d + 'area',
                                 'emitter' + d + 'pulse' + d + 'delay',
                                 'emitter' + d + 'pulse' + d + 'flip' + d + 'time',
                                 'emitter' + d + 'pulse' + d + 'phonon_coefficient',
                                 'emitter' + d + 'pulse' + d + 'width']
    assert system.default_parameters == {'atom' + d + 'decay': 1,
                                         'atom' + d + 'detuning': 0,
                                         'emitter' + d + 'decay' + d + 'rate': 1,
                                         'emitter' + d + 'dephasing' + d + 'rate': 0,
                                         'emitter' + d + 'detuning': 0,
                                         'emitter' + d + 'pulse' + d + 'area': 3.141592653589793,
                                         'emitter' + d + 'pulse' + d + 'delay': 0,
                                         'emitter' + d + 'pulse' + d + 'flip' + d + 'time': 1,
                                         'emitter' + d + 'pulse' + d + 'phonon_coefficient': 0.001,
                                         'emitter' + d + 'pulse' + d + 'width': 1}
    assert system.subdims == [2, 2]
    assert system.dim == 4
    assert system.bodies == 2
    assert system.has_instant
    assert system.has_interval
    assert system.is_time_dependent(0)
    assert system.is_nonhermitian_time_dependent(0)
    assert system.times() == [-6, 1, 6]
    assert system.support(0)
    assert not system.is_dirac(0)
    assert system.is_dirac(1)


def test_system_collection_evaluate():
    system = _make_controlled_twobody_system()

    x = create(2) + destroy(2)
    par = {'emitter' + d + 'detuning': 3, 'atom' + d + 'detuning': 2, 'dephasing' + d + 'rate': 2}
    expected_hamiltonian = tensor(3 * num(2) + sqrt(pi / 8) * x, qeye(2)) + tensor(qeye(2), 2 * num(2))
    expected_environment = [tensor(destroy(2), qeye(2)), tensor(2 * num(2), qeye(2)),
                            tensor(10e-4 * pi / 2 * num(2), qeye(2)), tensor(qeye(2), destroy(2))]
    assert system.evaluate(0, par) == liouvillian(H=expected_hamiltonian, c_ops=expected_environment)
    assert system.evaluate_dirac(1, {'flip time': 1}).evaluate() == \
           unitary_propagation_superoperator(tensor(create(2) + destroy(2), qeye(2)))
    assert system.evaluate_quadruple(0, par).hamiltonian.evaluate(0, par) == expected_hamiltonian
    assert [env.evaluate(0, par) for env in system.evaluate_quadruple(0, par).environment] == expected_environment
    assert system.evaluate_quadruple(0, par).evaluate(0, par) == system.evaluate(0, par)


def test_system_base_init_empty():
    system = MultiBodyEmitter()
    assert system.bodies == 0
    assert system.subsystems == {}
    assert system.coupling.evaluate(0) == Qobj()
    assert system.subdims is None
    assert system.dim is None
    assert system.evaluate(0) == Qobj()


def test_system_base_init():
    system = MultiBodyEmitter(subsystems=[_make_system()] * 3)
    system.subsystems['system 0'].add(HamiltonianBase(num(2)))
    assert system.dim == 2 ** 3
    assert system.subdims == [2] * 3
    assert system.bodies == 3
    assert not system.is_time_dependent(0)
    assert system.evaluate(0) == liouvillian(H=tensor(num(2), qeye(2), qeye(2)) +
                                               tensor(qeye(2), num(2), qeye(2)) +
                                               tensor(qeye(2), qeye(2), num(2)),
                                             c_ops=[tensor(destroy(2), qeye(2), qeye(2)),
                                                    tensor(qeye(2), destroy(2), qeye(2)),
                                                    tensor(qeye(2), qeye(2), destroy(2))])
    assert system.coupling.evaluate(0) == Qobj()
    assert system.coupling.dim is None
    assert system.coupling.subdims is None


def test_system_base_coupling():
    system = MultiBodyEmitter(subsystems=[_make_system()] * 2)
    system.add(_make_jaynes_cummings())
    assert system.dim == 4
    assert system.subdims == [2, 2]
    assert system.coupling.evaluate(0) == liouvillian(tensor(destroy(2), create(2)) + tensor(create(2), destroy(2)))
    assert system.evaluate(0) == liouvillian(H=tensor(destroy(2), create(2)) + tensor(create(2), destroy(2)),
                                             c_ops=[tensor(destroy(2), qeye(2)), tensor(qeye(2), destroy(2))])
