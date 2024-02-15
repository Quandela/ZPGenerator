from zpgenerator.time.vector import *
from zpgenerator.time.operator import *
from zpgenerator.time.function import *
from zpgenerator.time.evaluate.dirac import unitary_propagation_superoperator
from zpgenerator.time.pulse import PulseBase
from zpgenerator.time.parameters import Parameters
from qutip import destroy, create, qzero, spre, spost, sprepost
from pytest import raises

d = Parameters.DELIMITER


def test_time_operator_init_list_default():
    matrix = [[1, 2], [3, 4]]
    op = TimeOperator(operator=matrix)
    assert op.partial_evaluate(0).constant == Qobj(matrix)
    assert op.partial_evaluate(0).variable == []
    assert op.evaluate(0) == Qobj(matrix)
    assert op.support(0)
    assert op.dim == 2
    assert op.subdims == [2]
    assert not op.is_super
    assert not op.is_time_dependent(0)
    assert op.evaluate_function(0) == 1
    assert op.evaluate_dirac(0).evaluate() == sprepost(qeye(2), qeye(2))


def test_time_operator_init_callable():
    op = TimeOperator(operator=lambda args: [[args['a'], 2], [3, 4]], parameters={'a': 1})
    op.add(2)
    assert op.partial_evaluate(2).constant == 2 * Qobj([[1, 2], [3, 4]])
    op.add(lambda t, args: args['b'] * t ** 2, {'b': 2})
    assert op.partial_evaluate(2).constant == 0 * Qobj([[1, 2], [3, 4]])
    assert op.partial_evaluate(2).variable[0].evaluate(2) == 10 * Qobj([[1, 2], [3, 4]])
    assert op.partial_evaluate(2, {'a': 2, 'b': 3}).variable[0].evaluate(4, {'b': 3}) == 50 * Qobj([[2, 2], [3, 4]])
    assert op.evaluate(2) == op.partial_evaluate(2).variable[0].evaluate(2)


def test_time_operator_init_time_function():
    func = CompositeTimeFunction()
    func.add(lambda t, args: args['a'] * t ** 2, {'a': 1})
    op = TimeOperator(operator=destroy(2), functions=func)
    assert op.partial_evaluate(2).constant == qzero(2)
    assert op.evaluate(2) == 4 * destroy(2)
    assert op.parameters == ['a']
    assert op.parameter_tree() == {'_CompositeTimeFunction': {'_TimeFunction': {'a': 1}}}


def test_time_operator_parameters():
    func = CompositeTimeFunction()
    func.add(lambda t, args: args['a'] * t ** 2, {'a': 1})
    op = TimeOperator(operator=lambda args: args['b'] * destroy(2), functions=func, parameters={'b': 2})
    op.add(lambda t, args: args['c'] * t, parameters={'c': 2})
    assert op.partial_evaluate(2).constant == qzero(2)
    assert op.evaluate(2) == 16 * destroy(2)
    assert op.parameters == ['a', 'b', 'c']
    assert op.parameter_tree() == {'_CompositeTimeFunction': {'_TimeFunction': {'a': 1}},
                                   '_Operator': {'b': 2},
                                   '_TimeFunction': {'c': 2}}


def test_time_operator_pulse_compose():
    func = PulseBase(functions=TimeFunction(lambda t, args: args['a'] * t ** 2), parameters={'a': 2})
    func.compose_with(lambda amp, args: amp ** args['a'])
    op = TimeOperator(operator=destroy(2), functions=func)
    assert op.evaluate(2) == 64 * destroy(2)
    assert op.evaluate(2, {'a': 3}) == 1728 * destroy(2)


def test_time_operator_pulse_compose_composite():
    func = PulseBase(functions=TimeFunction(lambda t, args: args['a'] * t ** 2), parameters={'a': 2})
    func.compose_with(lambda amp, args: amp ** args['a'])
    op = TimeOperator(operator=destroy(2), functions=func)
    comp = TimeVectorOperator()
    comp.add(op)
    assert comp.evaluate(2)[0] == 64 * destroy(2)
    assert comp.evaluate(2, {'a': 3})[0] == 1728 * destroy(2)


def test_time_operator_with_dirac():
    func = TimeInstantFunction(value=2, instant='a', parameters={'a': 1})
    op = TimeOperator(operator=destroy(2) + create(2), functions=func)
    op.add(TimeIntervalFunction(value=lambda t, args: t ** 2, interval=[0.5, 1.5]))
    assert op.has_instant
    assert op.has_interval
    assert not op.support(0)
    assert not op.is_dirac(0)
    assert not op.is_time_dependent(0)
    assert op.support(1)
    assert op.is_dirac(1)
    assert not op.is_dirac(0.75)
    assert op.is_time_dependent(1)
    assert not op.is_time_dependent(2, {'a': 2})
    assert op.evaluate(0) == qzero(2)
    assert op.evaluate(1) == destroy(2) + create(2)
    assert op.evaluate_function(1)(1) == 1
    assert op.evaluate_dirac(0).evaluate() == sprepost(qeye(2), qeye(2))
    assert op.evaluate_dirac(1).evaluate() == unitary_propagation_superoperator(2 * (destroy(2) + create(2)))


def test_time_operator_collection_init_empty():
    comp_op = TimeOperatorCollection()
    assert comp_op.partial_evaluate(0).constant == Qobj()
    assert not comp_op.is_time_dependent(0)
    assert not comp_op.support(0)
    assert comp_op.operators == comp_op.functions


def test_time_operator_collection_init():
    op = TimeOperator(operator=destroy(2))
    assert op.evaluate(0) == destroy(2)
    comp_op = TimeOperatorCollection(operators=op)
    assert comp_op.partial_evaluate(0).constant == destroy(2)
    assert comp_op.evaluate(0) == destroy(2)
    assert not comp_op.is_time_dependent(0)
    assert not comp_op.is_super
    assert not comp_op.is_dirac(0)


def test_time_operator_collection_add_basic():
    op = TimeOperatorCollection()

    op.add(2)
    assert op.evaluate(0) == Qobj(2)

    op.add([[2.5]])
    assert op.evaluate(0) == Qobj(4.5)

    op.add(Qobj(3.j))
    assert op.evaluate(0) == Qobj(4.5 + 3.j)

    op.add(lambda t, args: t ** 2)
    assert op.evaluate(2) == Qobj(8.5 + 3.j)

    with raises(Exception):
        op.add(Qobj([[1, 2], [3, 4]]))

    op = TimeOperatorCollection()
    op.add(Qobj([[1, 2], [3, 4]]))
    assert op.evaluate(0) == Qobj([[1, 2], [3, 4]])

    op.add(lambda args: [[args['a'], 1], [1, 1]], parameters={'a': 1})
    assert op.evaluate(0) == Qobj([[2, 3], [4, 5]])
    assert op.evaluate(0, {'a': 2}) == Qobj([[3, 3], [4, 5]])

    with raises(Exception):
        op.add(lambda t, args: t ** 2)


def test_time_operator_collection_add_advanced():
    op = TimeOperator(operator=destroy(2))
    op.add(lambda t, args: args['a'] * t ** 2, parameters={'a': 5})
    assert op.parameters == ['a']
    comp_op = TimeOperatorCollection()
    comp_op.add(op)
    comp_op.add(lambda args: args['b'] * create(2), parameters={'b': 2})
    assert comp_op.evaluate(2) == 20 * destroy(2) + 2 * create(2)

    comp_op.add(TimeOperator(operator=lambda args: args['b'] * create(2),
                             functions=TimeIntervalFunction(value=lambda t, args: args['c'] * t ** -1,
                                                            interval=[0, 'd'],
                                                            parameters={'c': 2, 'd': 1}),
                             parameters={'b': 3}))
    assert comp_op.evaluate(2) == 20 * destroy(2) + 2 * create(2)
    assert comp_op.evaluate(2, {'d': 3}) == 20 * destroy(2) + 5 * create(2)
    assert comp_op.evaluate(2, {'d': 3, 'b': 1}) == 20 * destroy(2) + 2 * create(2)
    assert comp_op.evaluate(2, {'a': 2, 'd': 1, 'b': 1}) == 8 * destroy(2) + create(2)
    assert comp_op.parameters == ['a', 'b', 'c', 'd']
    assert comp_op.parameter_tree() == {'_TimeOperator': {'_TimeFunction': {'a': 5}},
                                        '_TimeOperator (1)': {'_operator': {'b': 2}},
                                        '_TimeOperator (2)': {'_TimeIntervalFunction': {'_default': {'c': 2, 'd': 1},
                                                                                        '_interval': {'c': 2, 'd': 1},
                                                                                        '_value': {'c': 2, 'd': 1}},
                                                              '_operator': {'b': 3}}}


def test_time_operator_collection_dirac():
    comp_op = TimeOperatorCollection()
    comp_op.add(TimeOperator.dirac(destroy(2), 2))
    assert comp_op.is_dirac(2)
    assert comp_op.evaluate_dirac(2).evaluate() == unitary_propagation_superoperator(destroy(2))
    assert comp_op.evaluate(2) == qzero(2)

    comp_op.add(TimeOperator.dirac(create(2), 'time', parameters={'time': 3}, name='now'))
    assert comp_op.evaluate_dirac(3).evaluate() == unitary_propagation_superoperator(create(2))
    assert comp_op.evaluate_dirac(2, parameters={'time': 2}).evaluate() == \
           unitary_propagation_superoperator(create(2) + destroy(2))
    assert comp_op.parameters == ['now' + d + 'time']
    assert comp_op.default_parameters == {'now' + d + 'time': 3}
    assert comp_op.parameter_tree() == {'now': {'_TimeInstantFunction': {'_value (1)': {'time': 3}},
                                                '_operator': {'time': 3}}}


def test_composite_time_operator_init_empty():
    comp = CompositeTimeOperator()
    assert comp.operators == {}
    assert comp.evaluate(0) == 0
    assert comp.dim is None
    assert comp.subdims is None


def test_composite_time_operator_init():
    comp = CompositeTimeOperator(operators=TimeOperator(destroy(3),
                                                        functions=TimeFunction(lambda t, args: args['a'] * t ** 2,
                                                                               parameters={'a': 2})))
    assert comp.evaluate(0) == qzero(3)
    assert comp.evaluate(2) == 8 * destroy(3)
    with raises(Exception):
        comp.add(destroy(4))


def test_composite_time_operator_add():
    comp = CompositeTimeOperator()
    comp.add(spre(destroy(2)))
    op = TimeOperator(spost(create(2)))
    op.add(lambda t, args: args['a'] * t ** 2, parameters={'a': 2})
    comp.add(op)
    assert comp.evaluate(2) == spre(destroy(2)) + 8 * spost(create(2))


def test_composite_time_operator_add_parameters():
    comp = CompositeTimeOperator()
    comp.add(lambda args: args['a'] * destroy(2), parameters={'a': 2})
    assert comp.parameters == ['a']

    comp = CompositeTimeOperator()
    comp.add(lambda args: args['a'] * destroy(3), parameters={'a': 2}, name='destroy')
    assert comp.parameters == ['destroy' + d + 'a']
    assert comp.parameter_tree() == {'_TimeOperator': {'destroy': {'a': 2}}}


def test_time_vector_operator_init_empty():
    vec = TimeVectorOperator()
    assert vec.subdims is None
    assert vec.dim is None
    assert vec.length == 0
    assert vec.evaluate(0) == []


def test_time_vector_operator_init():
    vec = TimeVectorOperator(operators=TimeOperator([[1, 2], [3, 4]], functions=TimeFunction(lambda t, args: t ** 2)))
    assert vec.subdims == [2]
    assert vec.dim == 2
    assert vec.length == 1
    assert vec.evaluate(2) == [4 * Qobj([[1, 2], [3, 4]])]


def test_time_vector_operator_add():
    vec = TimeVectorOperator()
    vec.add(lambda t, args: t ** 2)
    vec.add(lambda t, args: t ** 3)
    assert vec.evaluate(2) == [Qobj([[4]]), Qobj([[8]])]
    with raises(Exception):
        vec.add(destroy(2))

    vec = TimeVectorOperator()
    vec.add(destroy(2))
    vec.add(lambda args: args['a'] * create(2), parameters={'a': 1})
    vec.add(TimeOperator(create(2) * destroy(2), TimeFunction(lambda t, args: args['b'] * t ** 2, parameters={'b': 1})))
    assert vec.length == 3
    assert vec.subdims == [2]
    assert vec.evaluate(2) == [destroy(2), create(2), 4 * create(2) * destroy(2)]
    assert vec.evaluate(2, {'a': 2, 'b': 2}) == [destroy(2), 2 * create(2), 8 * create(2) * destroy(2)]


def test_time_vector_add_parameters():
    vec = TimeVectorOperator()
    vec.add(lambda args: args['a'] * destroy(2), parameters={'a': 2}, name='test')
    assert vec.parameters == ['test' + d + 'a']
    assert vec.parameter_tree() == {'_TimeOperator': {'test': {'a': 2}}}
