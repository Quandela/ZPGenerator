from zpgenerator.system.scatterer import *
from zpgenerator.time import Operator, TimeOperator, TimeFunction
from qutip import Qobj, qeye, qzero, create, destroy, num


def test_scatterer_constant():
    s = ScattererBase()
    s.add(Operator([[1, 1], [1, 1]]))
    assert s.evaluate(0) == Qobj([[1, 1], [1, 1]])

    s.add(qeye(2))
    assert s.evaluate(0) == Qobj([[2, 1], [1, 2]])

    assert s.modes == 2

    s = ScattererBase()
    s.add(qeye(5))
    assert s.modes == 5


def test_multiscatterer_constant():
    s = MultiScatterer()
    s.add(Operator([[1, 1], [1, 1]]))
    s.add(Operator([[0, 1], [0, 0]]))
    assert s.evaluate(0) == Qobj([[0, 1], [0, 1]])
    assert s.partial_evaluate(0).constant == Qobj([[0, 1], [0, 1]])


def test_multiscatterer_time_dependent():
    s = MultiScatterer()
    s.add(TimeOperator(num(3), TimeFunction(lambda t, args: args['a'] * t ** 2, parameters={'a': 1})))
    s.add(TimeOperator(create(3), TimeFunction(lambda t, args: args['b'] * t ** -1, parameters={'b': 1})))
    assert s.evaluate(2, {'a': 2, 'b': 3}) == (8 * num(3)) * (3 / 2 * create(3))
    assert s.partial_evaluate(2).constant == qzero(3)
    assert s.partial_evaluate(2).variable[0].list_form()[0] == num(3) * create(3)

    s = ScattererBase(matrices=s)
    s.add(destroy(3))
    assert s.evaluate(2, {'a': 2, 'b': 3}) == (8 * num(3)) * (3 / 2 * create(3)) + destroy(3)


def test_complicated_time_dependent_scatterer():
    s00 = TimeOperator(destroy(2), TimeFunction(lambda t, args: args['a'] * t ** 2, parameters={'a': 2}))
    s01 = TimeOperator(num(2), TimeFunction(lambda t, args: args['b'] * t ** 3, parameters={'b': 1 / 4}))
    s10 = TimeOperator(create(2), TimeFunction(lambda t, args: args['c'] * t ** -1, parameters={'c': 4}))
    s11 = TimeOperator(destroy(2) * create(2), TimeFunction(lambda t, args: args['d'] * t ** -2, parameters={'d': 1}))

    s = ScattererBase()
    s.add([s00, s01, s10, s11])
    assert s.evaluate(2) == Qobj([[1/4, 8], [2, 2]])

    s00 = TimeOperator(destroy(2), TimeFunction(lambda t, args: args['a'] * t ** -2, parameters={'a': 2}))
    s01 = TimeOperator(num(2), TimeFunction(lambda t, args: args['b'] * t ** 4, parameters={'b': 1}))
    s10 = TimeOperator(create(2), TimeFunction(lambda t, args: args['c'] * t, parameters={'c': 1}))
    s11 = TimeOperator(destroy(2) * create(2), TimeFunction(lambda t, args: args['d'] * t ** -3, parameters={'d': 1}))

    m = ScattererBase()
    m.add([s00, s01, s10, s11])

    s = MultiScatterer(matrices=[s, m])
    s.add(lambda args: args['e'] * destroy(2), parameters={'e': 2})
    assert s.evaluate(2, {'a': 1, 'e': 5}) == 5 * Qobj([[1/4, 4], [2, 2]]) * Qobj([[1/8, 1/4], [2, 16]]) * destroy(2)


def test_haar_random():
    rand = ScattererBase.haar_random(4)
    mat = rand.evaluate(0)
    assert mat * mat.dag() == qeye(4)
