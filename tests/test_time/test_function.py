from zpgenerator.time.function import *


def test_value_constant():
    value = TimeFunction(value=1)
    assert value.value == 1
    assert not value.is_callback
    assert not value.is_time_dependent(0)
    assert value.support(0)
    assert value.times() == []
    assert value.evaluate(10) == 1
    assert not value.is_dirac(10)


def test_value_callable():
    value = TimeFunction(value=lambda args: args['a'], parameters={'a': 1})
    assert value.value({'a': 2}) == 2
    assert value.is_callback
    assert not value.is_time_dependent(0)
    assert value.support(0)
    assert value.times() == []
    assert value.evaluate(10) == 1
    assert not value.is_dirac(10)

    value = TimeFunction(value=lambda t, args: args['a'] * t ** 2, parameters={'a': 1})
    assert value.value(2, {'a': 2}) == 8
    assert value.is_callback
    assert value.is_time_dependent(0)
    assert value.support(0)
    assert value.times() == []
    assert value.evaluate(10) == 100
    assert not value.is_dirac(10)


def test_interval_function_constant():
    func = TimeIntervalFunction(value=1)
    assert func.value.value == 1
    assert func.interval.interval == [float('-inf'), float('inf')]
    assert func.default.value == 0
    assert not func.freeze
    assert func.support(0)
    assert func.times() == []
    assert func.evaluate(0) == 1
    assert not func.is_time_dependent(0)
    assert not func.is_dirac(0)


def test_interval_function_square():
    func = TimeIntervalFunction(value=1, interval=[-1, 1])
    assert func.value.value == 1
    assert func.interval.interval == [-1, 1]
    assert func.default.value == 0
    assert not func.freeze
    assert func.support(0)
    assert not func.support(2)
    assert func.times() == [-1, 1]
    assert func.evaluate(0) == 1
    assert func.evaluate(2) == 0
    assert not func.is_time_dependent(0)
    assert not func.is_dirac(0)


def test_interval_function_time_dependent():
    func = TimeIntervalFunction(value=lambda t, args: t, interval=[-1, 1])
    assert func.value.value(0, {}) == 0
    assert func.evaluate(-1.01) == 0
    assert func.evaluate(-1) == -1
    assert func.evaluate(0.99) == 0.99
    assert func.evaluate(1) == 0
    assert func.evaluate(2) == 0
    assert func.is_time_dependent(0)
    assert not func.is_time_dependent(2)
    assert not func.is_dirac(0)


def test_interval_function_freeze():
    func = TimeIntervalFunction(value=lambda t, args: args['a'] * t**2,
                                interval=[-1, 'b'], freeze=True,
                                parameters={'a': 1, 'b': 1})
    assert func.evaluate(-2) == 1
    assert func.evaluate(-1) == 1
    assert func.evaluate(0) == 0
    assert func.evaluate(1) == 1
    assert func.evaluate(2) == 1
    assert func.evaluate(2, {'b': 2}) == 4
    assert func.evaluate(3, {'b': 2}) == 4
    assert func.evaluate(3, {'a': 2, 'b': 2}) == 8


def test_instant_function_constant():
    func = TimeInstantFunction(value=1, instant=0)
    assert func.evaluate(0) == 0
    assert func.evaluate(1) == 0
    assert func.evaluate_dirac(0) == 1
    assert func.evaluate_dirac(1) == 0
    assert func.support(0)
    assert not func.support(1)
    assert func.times() == [0]
    assert not func.is_time_dependent(0)
    assert func.is_dirac(0)
    assert not func.is_dirac(1)


def test_instant_function_parameterized():
    func = TimeInstantFunction(value=lambda args: args['a'], instant='b',
                               parameters={'a': 1, 'b': 0}, default=3)
    assert func.evaluate(1, {'b': 1}) == 0
    assert func.evaluate(2, {'a': 2, 'b': 2}) == 0
    assert func.evaluate(3, {'b': 1}) == 0
    assert func.evaluate_dirac(1, {'b': 1}) == 1
    assert func.evaluate_dirac(2, {'a': 2, 'b': 2}) == 2
    assert func.evaluate_dirac(3, {'b': 1}) == 3
    assert func.is_dirac(3, {'b': 3})
    assert not func.is_dirac(3, {'b': 2})
