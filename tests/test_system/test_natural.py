from zpgenerator.system.natural import *
from zpgenerator.time import Operator, TimeOperator, TimeIntervalFunction, TimeInterval, TimeFunction, \
    unitary_propagation_superoperator
from qutip import Qobj, num, destroy, create, sprepost, qzero, liouvillian, qeye
from numpy import pi, exp, sqrt
from zpgenerator.time.parameters import Parameters

d = Parameters.DELIMITER

def test_hamiltonian_init_empty():
    ham = HamiltonianBase()
    assert ham.subdims is None
    assert ham.operators == {}
    assert ham.evaluate(0) == Qobj()


def test_hamiltonian_init_basic():
    ham = HamiltonianBase(operators=Operator(Qobj(num(3))))
    assert ham.subdims == [3]
    assert ham.operators['_TimeOperator'].evaluate(0) == num(3)
    assert ham.evaluate(0) == liouvillian(num(3))


def test_hamiltonian_add():
    op = TimeOperator(num(3))
    op.add(lambda t, args: args['a'] * t ** 2, parameters={'a': 1})
    ham = HamiltonianBase(operators=op)
    ham.add(destroy(3) + create(3))
    ham.add(lambda args: args['omega'] * (destroy(3) + create(3)) / 2, parameters={'omega': 0})
    assert ham.uses_parameter('a')
    assert ham.evaluate(2, parameters={'a': 2}) == liouvillian(8 * num(3) + destroy(3) + create(3))
    assert ham.evaluate_quadruple(2, parameters={'a': 2}).hamiltonian.constant == destroy(3) + create(3)
    assert ham.evaluate_quadruple(2, parameters={'omega': 2}).hamiltonian.constant == 2 * (destroy(3) + create(3))
    assert ham.parameter_tree() == {'_TimeOperator': {'_TimeFunction': {'a': 1}},
                                    '_TimeOperator (2)': {'_operator': {'omega': 0}}}


def test_hamiltonian_add_direct():
    ham0 = HamiltonianBase(destroy(2))
    ham1 = HamiltonianBase(create(2))
    ham3 = ham0 + ham1
    assert ham3.evaluate(0) == liouvillian(destroy(2) + create(2))


def test_hamiltonian_quantum_system():
    ham = HamiltonianBase()

    # manual construction of a two-level system driven by a Gaussian pulse
    ham.add(lambda args: args['delta'] * create(2) * destroy(2), parameters={'delta': 0})
    ham.add(TimeOperator(operator=(destroy(2) + create(2)) / 2,
                         functions=TimeIntervalFunction(
                             value=lambda t, args: args['area'] * exp(-t ** 2 / (2 * args['width'] ** 2)) / sqrt(
                                 2 * pi * args['width'] ** 2),
                             interval=TimeInterval(lambda args: [args['delay'] - 6 * args['width'],
                                                                 args['delay'] + 6 * args['width']],
                                                   parameters={'delay': 0, 'width': 1}),
                             parameters={'area': pi, 'width': 1})))

    assert not ham.is_nonhermitian_time_dependent(0)
    assert ham.evaluate_quadruple(0, {'delta': 2}).hamiltonian.constant == 2 * create(2) * destroy(2)
    assert ham.evaluate(0, {'delta': 2}) == liouvillian(2 * create(2) * destroy(2) +
                                                        sqrt(pi / 8) * (create(2) + destroy(2)))


def test_hamiltonian_add_parameters():
    ham = HamiltonianBase()
    ham.add(lambda args: args['a'] * destroy(2), parameters={'a': 2})
    assert ham.parameters == ['a']


def test_environment_init_empty():
    env = EnvironmentBase()
    assert env.subdims is None
    assert env.operators == {}
    assert env.evaluate(0) == Qobj()


def test_environment_init():
    env = EnvironmentBase(operators=[TimeOperator(lambda args: args['jump'] * sprepost(destroy(2), create(2)),
                                                  functions=TimeIntervalFunction(lambda t, args: t ** 2, interval=[-1, 2]),
                                                  parameters={'jump': 0}),
                                     TimeOperator(lambda args: args['dephasing'] * num(2),
                                              functions=TimeFunction(lambda t, args: t ** 3),
                                              parameters={'dephasing': 0})])
    assert env.subdims == [2]
    assert env.length == 2
    assert env.is_super
    assert env.evaluate(1, parameters={'jump': 1, 'dephasing': 2}) == \
           liouvillian(qzero(2), [sprepost(destroy(2), create(2)), 2 * num(2)])
    assert env.evaluate_quadruple(0, parameters={'jump': 1, 'dephasing': 2}).environment[0].constant == \
           sprepost(qzero(2), qzero(2))
    assert env.evaluate_quadruple(0, parameters={'jump': 1, 'dephasing': 2}).environment[1].constant == qzero(2)
    assert env.evaluate_quadruple(2).environment[0].variable == []
    assert env.evaluate_quadruple(2, parameters={'dephasing': 1}).environment[1].variable[0].op == num(2)


def test_environment_add_direct():
    env0 = EnvironmentBase(destroy(2))
    env1 = EnvironmentBase(create(2))
    env2 = env0 + env1
    assert env2.evaluate(0) == liouvillian(qzero(2), [destroy(2), create(2)])


def test_natural_system_init_empty():
    sys = NaturalSystem()
    assert sys.hamiltonian.evaluate(0) == Qobj()
    assert sys.environment.evaluate(0) == Qobj()
    assert sys.states == {}
    assert sys.operators == {}
    assert sys.evaluate(0) == Qobj()


def _make_natural_system():
    ham = HamiltonianBase()
    ham.add(lambda args: args['detuning'] * create(2) * destroy(2), parameters={'detuning': 0})
    env = EnvironmentBase()
    env.add(lambda args: args['decay'] * destroy(2), parameters={'decay': 1})
    return NaturalSystem(hamiltonian=ham, environment=env, name='emitter')


def test_natural_system_init():
    sys = _make_natural_system()
    assert not sys.is_super
    assert sys.dim == 2
    assert sys.subdims == [2]
    assert sys.evaluate(0) == liouvillian(H=qzero(2), c_ops=[destroy(2)])
    assert sys.evaluate_quadruple(0, {'detuning': 1}).hamiltonian.constant == create(2) * destroy(2)


def test_natural_system_add():
    ham = HamiltonianBase()
    ham.add(create(3) * destroy(3))
    sys = NaturalSystem(hamiltonian=ham, name='Sys')
    sys.add(EnvironmentBase(TimeOperator(destroy(3))))
    sys.add(EnvironmentBase(TimeOperator(operator=destroy(3),
                                         functions=TimeIntervalFunction(value=lambda t, args: args['a'] * t ** 2,
                                                                    interval=[-1, 3],
                                                                    parameters={'a': 1},
                                                                    name='quadratic'))))
    assert sys.evaluate(0) == liouvillian(H=create(3) * destroy(3), c_ops=[destroy(3)])
    assert sys.evaluate(2, {'a': 2}) == liouvillian(H=create(3) * destroy(3), c_ops=[destroy(3), 8 * destroy(3)])
    assert sys.is_nonhermitian_time_dependent(0)
    assert sys.parameter_tree() == {'_environment': {'_EnvironmentBase (1)': {'_TimeOperator': {'quadratic': {
        '_default': {'a': 1},
        '_interval': {'a': 1},
        '_value': {'a': 1}}}}}}
    sys.hamiltonian.add(lambda args: args['Omega'] * (destroy(3) + create(3)) / 2,
                        parameters={'Omega': pi}, name='Rabi')

    assert sys.evaluate(2, {'a': 2}) == liouvillian(H=create(3) * destroy(3) + pi * (destroy(3) + create(3)) / 2,
                                                    c_ops=[destroy(3), 8 * destroy(3)])
    assert sys.parameter_tree()['_hamiltonian'] == {'_TimeOperator (1)': {'Rabi': {'Omega': pi}}}
    assert sys.hamiltonian.parameters == ['Rabi' + d + 'Omega']


def test_natural_system_dirac():
    sys = NaturalSystem()
    sys.hamiltonian.add(TimeOperator.dirac(destroy(2) + create(2), time=2))
    assert sys.evaluate_dirac(0).evaluate() == sprepost(qeye(2), qeye(2))
    assert sys.evaluate_dirac(2).evaluate() == unitary_propagation_superoperator(destroy(2) + create(2))
