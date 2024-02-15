from zpgenerator.system.coupling import *
from zpgenerator.time import TimeFunction, Operator
from qutip import create, destroy, tensor, num, qeye, qzero, liouvillian
from numpy import exp


def test_coupling_term_constant():
    c = CouplingTerm()
    c.add(destroy(2))
    c.add(create(2))
    assert c.evaluate(0) == tensor(destroy(2), create(2))
    assert c.bodies == 2
    assert c.subdims == [2, 2]
    assert c.dim == 4

    c.add(num(3))
    assert c.evaluate(0) == tensor(destroy(2), create(2), num(3))
    assert c.bodies == 3
    assert c.subdims == [2, 2, 3]
    assert c.dim == 12


def test_coupling_term_tensor_insert():
    c = CouplingTerm(operators=[destroy(2), create(2), num(3)])
    c.insert_dimension(1, 3)
    assert c.evaluate(0) == tensor(destroy(2), qeye(3), create(2), num(3))
    assert c.subdims == [2, 3, 2, 3]
    c.insert_dimension(3, [2, 2])
    assert c.evaluate(0) == tensor(destroy(2), qeye(3), create(2), qeye([2, 2]), num(3))
    assert c.subdims == [2, 3, 2, 2, 2, 3]


def test_coupling_term_pad():
    c = CouplingTerm(operators=[destroy(2), create(3)])
    c.pad_left(6)
    assert c.evaluate(0) == tensor(qeye(6), destroy(2), create(3))
    assert c.subdims == [6, 2, 3]
    c.pad_right(4)
    assert c.evaluate(0) == tensor(qeye(6), destroy(2), create(3), qeye(4))
    assert c.subdims == [6, 2, 3, 4]


def test_coupling_term_partial_evaluate_constant():
    c = CouplingTerm(operators=[destroy(2), create(3)])
    assert c.partial_evaluate(0).constant == tensor(destroy(2), create(3))
    assert c.partial_evaluate(0).variable == []


def test_coupling_term_time_dependent():
    c = CouplingTerm()
    c.add(TimeOperator(destroy(2), functions=TimeFunction(lambda t, args: t ** 2)))
    c.add(create(3))
    assert c.evaluate(2) == 4 * tensor(destroy(2), create(3))
    assert c.partial_evaluate(0).constant == qzero([2, 3])
    assert c.partial_evaluate(0).variable[0].op == tensor(destroy(2), create(3))
    assert c.partial_evaluate(0).variable[0].func(2) == 4


def test_coupling_base_constant():
    jc = CouplingBase()
    jc.add(CouplingTerm(operators=[Operator(lambda args: args['g'] * destroy(2), parameters={'g': 1}), create(3)]))
    jc.add(CouplingTerm(operators=[Operator(lambda args: args['g'] * create(2), parameters={'g': 1}), destroy(3)]))
    jc.name = 'JC'
    assert jc.evaluate(0, {'g': 2}) == liouvillian(2 * (tensor(destroy(2), create(3)) + tensor(create(2), destroy(3))))
    assert jc.parameters == ['g']
    assert jc.default_parameters == {'g': 1}
    assert jc.parameter_tree() == {'_CouplingTerm': {'_TimeOperator': {'_operator': {'g': 1}}},
                                   '_CouplingTerm (1)': {'_TimeOperator': {'_operator': {'g': 1}}}}
    assert jc.bodies == 2
    assert not jc.is_time_dependent(0)
    assert jc.subdims == [2, 3]


def _make_jaynes_cummings(n=2, m=2):
    jc = CouplingBase()
    jc.add(CouplingTerm(operators=[TimeOperator(destroy(n),
                                                TimeFunction(lambda t, args: args['g'] * exp(1.j * args['omega'] * t),
                                                             parameters={'g': 1, 'omega': 0})),
                                   create(m)]))
    jc.add(CouplingTerm(operators=[TimeOperator(create(n),
                                                TimeFunction(lambda t, args: args['g'] * exp(-1.j * args['omega'] * t),
                                                             parameters={'g': 1, 'omega': 0})),
                                   destroy(m)]))
    jc.name = 'JC'
    return jc


def test_coupling_base_time_dependent():
    jc = _make_jaynes_cummings(2, 3)
    assert jc.evaluate(1, {'g': 2, 'omega': 2}) == liouvillian(2 * (tensor(destroy(2), create(3)) * exp(2.j) +
                                                                    tensor(create(2), destroy(3)) * exp(-2.j)))
    assert jc.default_parameters == {'g': 1, 'omega': 0}
    assert jc.parameter_tree() == {'_CouplingTerm': {'_TimeOperator': {'_TimeFunction': {'g': 1, 'omega': 0}}},
                                   '_CouplingTerm (1)': {'_TimeOperator': {'_TimeFunction': {'g': 1, 'omega': 0}}}}
    assert jc.bodies == 2
    assert jc.is_time_dependent(0)
    assert jc.subdims == [2, 3]
