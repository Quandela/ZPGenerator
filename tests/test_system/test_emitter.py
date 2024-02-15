from zpgenerator.system.emitter import *
from test_control import _make_controlled_system
from qutip import destroy, num, create
from zpgenerator.time.parameters import Parameters


d = Parameters.DELIMITER

def test_lindblad_vector():
    vec = LindbladVector()
    assert vec.modes == 0
    vec.add(destroy(2))
    assert vec.modes == 1
    vec.add(destroy(2))
    assert vec.modes == 2


def test_emitter_base_init_empty():
    emitter = EmitterBase()
    assert emitter.dim is None
    assert emitter.subdims is None
    assert emitter.parameters == []
    assert emitter.states == {}
    assert emitter.operators == {}
    assert emitter.evaluate(0) == Qobj()
    assert emitter.transitions.evaluate(0) == Qobj()
    assert emitter.modes == 0


def test_emitter_base_init():
    system = _make_controlled_system()
    emitter = EmitterBase(hamiltonian=system.hamiltonian,
                          environment=system.environment,
                          control=system.control,
                          transitions=[destroy(2), destroy(2)],
                          states=system.states,
                          operators=system.operators)

    assert emitter.modes == 2
    assert emitter.dim == 2
    assert emitter.subdims == [2]
    assert emitter.parameters == ['decay' + d + 'rate',
                                  'dephasing' + d + 'rate',
                                  'detuning',
                                  'pulse' + d + 'area',
                                  'pulse' + d + 'delay',
                                  'pulse' + d + 'flip' + d + 'time',
                                  'pulse' + d + 'phonon_coefficient',
                                  'pulse' + d + 'width']
    assert emitter.states == {}
    assert emitter.operators == {}
    assert emitter.evaluate(0) == system.evaluate(0)
    assert emitter.transitions.evaluate(0) == Qobj() # transitions do not have a Liouvillian
    assert emitter.evaluate_quadruple(0).transitions[0].constant == destroy(2)


def test_emitter_add():
    emitter = EmitterBase()
    env = EnvironmentBase()
    env.add(lambda args: args['a'] * destroy(2), parameters={'a': 1})
    emitter.add(env)

    # adding an EnvironmentBase directly to emitter nests it within the current environment
    assert emitter.parameter_tree() == {'_environment': {'_EnvironmentBase': {'_TimeOperator': {'_operator': {'a': 1}}}}}

    emitter = EmitterBase()
    emitter.environment.add(lambda args: args['a'] * destroy(2), parameters={'a': 1})

    # adding to the current emitter environment modifies the current environment
    assert emitter.parameter_tree() == {'_environment': {'_TimeOperator': {'_operator': {'a': 1}}}}

    # the same behaviour holds for HamiltonianBase, LindbladVector, and ControlBase
    emitter = EmitterBase()
    ham = HamiltonianBase()
    env = EnvironmentBase()
    trn = LindbladVector()
    con = ControlBase()
    ham.add(lambda args: args['a'] * num(2), parameters={'a': 0})
    env.add(lambda args: args['b'] * destroy(2), parameters={'b': 1})
    trn.add(lambda args: args['c'] * destroy(2), parameters={'c': 1})
    con.hamiltonian.add(lambda args: args['d'] * (destroy(2) + create(2)), parameters={'d': 2})

    emitter.add([ham, env, trn, con])
    assert emitter.parameter_tree() == {
        '_control': {'_ControlBase': {'_hamiltonian': {'_TimeOperator': {'_operator': {'d': 2}}}}},
        '_environment': {'_EnvironmentBase': {'_TimeOperator': {'_operator': {'b': 1}}}},
        '_hamiltonian': {'_HamiltonianBase': {'_TimeOperator': {'_operator': {'a': 0}}}},
        '_transitions': {'_LindbladVector': {'_TimeOperator': {'_operator': {'c': 1}}}}}
