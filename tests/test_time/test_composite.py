from zpgenerator.time.composite import *
from zpgenerator.time.function import *
from zpgenerator.time.parameters import Parameters

d = Parameters.DELIMITER

def test_time_function_collection_init_empty():
    func = TimeFunctionCollection()
    assert func.functions == {}
    assert func.evaluate(0) == 0
    assert func.evaluate_dirac(0) == 0
    assert func.times() == []
    assert not func.is_time_dependent(0)
    assert not func.is_dirac(0)


def test_time_function_collection_init_constant():
    func = TimeFunctionCollection(functions=1)
    assert func.support(0)
    assert func.evaluate(0) == 1
    assert func.evaluate_dirac(0) == 0
    assert func.times() == []
    assert not func.is_time_dependent(0)
    assert not func.is_dirac(0)


def test_time_function_collection_init_lambda():
    func = TimeFunctionCollection(functions=lambda t, args: args['a'] * t ** 2, parameters={'a': 1})
    assert func.support(0)
    assert func.evaluate(0) == 0
    assert func.evaluate(2) == 4
    assert func.evaluate(2, parameters={'a': 2}) == 8


def test_time_function_collection_init_interval_function():
    func0 = TimeIntervalFunction(value=lambda t, args: args['a'] * t ** 2,
                                 interval=['b', 'c'],
                                 parameters={'a': 1, 'b': 0, 'c': 1},
                                 name='quadratic')
    func = TimeFunctionCollection(functions=func0)
    assert func.support(0)
    assert not func.support(-1)
    assert func.evaluate(0) == 0
    assert func.evaluate(2, {'c': 2.01}) == 4
    assert func.evaluate(-1) == 0
    assert func.evaluate(-1, {'b': -1}) == 1
    assert func.parameters == ['quadratic' + d + 'a', 'quadratic' + d + 'b', 'quadratic' + d + 'c']
    assert func.default_parameters == {'quadratic' + d + 'a': 1, 'quadratic' + d + 'b': 0, 'quadratic' + d + 'c': 1}
    assert func.parameter_tree() == {'quadratic': {'_default': {'a': 1, 'b': 0, 'c': 1},
                                                   '_interval': {'a': 1, 'b': 0, 'c': 1},
                                                   '_value': {'a': 1, 'b': 0, 'c': 1}}}


def test_time_function_collection_init_interval_function_value_interval():
    func0 = TimeIntervalFunction(value=TimeFunction(value=lambda t, args: args['a'] * t ** 2, parameters={'a': 1}),
                                 interval=TimeInterval(interval=['b', 'c'], parameters={'b': 0, 'c': 1}),
                                 name='quadratic')
    func = TimeFunctionCollection(functions=func0)
    assert func.parameters == ['quadratic' + d + 'a', 'quadratic' + d + 'b', 'quadratic' + d + 'c']
    assert func.default_parameters == {'quadratic' + d + 'a': 1, 'quadratic' + d + 'b': 0, 'quadratic' + d + 'c': 1}
    assert func.functions['quadratic'].value.parameter_tree() == {'a': 1}
    assert func.functions['quadratic']._children.children[0].parameter_tree() == {'a': 1}
    assert func.parameter_tree() == {'quadratic': {'_interval': {'b': 0, 'c': 1},
                                                   '_value': {'a': 1}}}


def test_time_function_collection_add():
    func0 = TimeIntervalFunction(value=TimeFunction(value=lambda t, args: args['a'] * t ** 2, parameters={'a': 1}),
                                 interval=TimeInterval(interval=['b', 'c'], parameters={'b': 0, 'c': 4}),
                                 name='quadratic')
    func1 = TimeIntervalFunction(value=TimeFunction(value=lambda t, args: args['d'] * t ** -1, parameters={'d': 2}),
                                 interval=TimeInterval(interval=['e', 'f'], parameters={'e': -1, 'f': 3}),
                                 name='reciprocal')
    func2 = TimeFunctionCollection()
    func2.add(func0)
    func2.add(func1)
    assert func2.evaluate(2) == 5
    assert func2.evaluate(-1) == -2
    assert func2.support(-1)
    assert func2.support(3.99)
    assert not func2.support(4)
    assert func2.parameters == ['quadratic' + d + 'a',
                                'quadratic' + d + 'b',
                                'quadratic' + d + 'c',
                                'reciprocal' + d + 'd',
                                'reciprocal' + d + 'e',
                                'reciprocal' + d + 'f']
    assert func2.parameter_tree() == {'quadratic': {'_interval': {'b': 0, 'c': 4},
                                                    '_value': {'a': 1}},
                                      'reciprocal': {'_interval': {'e': -1, 'f': 3},
                                                     '_value': {'d': 2}}}
    assert func2.parameter_tree({'b': 1, 'd': 5}) == {'quadratic': {'_interval': {'b': 1, 'c': 4},
                                                                    '_value': {'a': 1}},
                                                      'reciprocal': {'_interval': {'e': -1, 'f': 3},
                                                                     '_value': {'d': 5}}}
    assert func2.times() == [-1, 0, 3, 4]
    assert func2.times({'b': 1}) == [-1, 1, 3, 4]
    assert func2.is_time_dependent(0)
    assert func2.is_time_dependent(-1)
    assert not func2.is_time_dependent(4)
    assert not func2.is_dirac(0)
    assert func2.evaluate_dirac(0) == 0


def test_time_function_collection_add_named_lambda():
    func = TimeFunctionCollection()
    func.add(lambda t, args: args['a'] * t**2, parameters={'a': 2}, name='quadratic')
    assert func.parameter_tree() == {'quadratic': {'a': 2}}
    assert func.evaluate(2, parameters={'a': 3}) == 12


def test_time_function_collection_instant():
    func0 = TimeIntervalFunction(value=TimeFunction(value=lambda t, args: args['a'] * t ** 2, parameters={'a': 1}),
                                 interval=TimeInterval(interval=['b', 'c'], parameters={'b': 0, 'c': 1}),
                                 name='quadratic')
    func1 = TimeInstantFunction(value=1, instant='d', name='switch', parameters={'d': 0.5})
    func2 = TimeFunctionCollection()
    func2.add([func0, func1])
    assert func2.is_dirac(0.5)
    assert not func2.is_dirac(1)
    assert func2.is_time_dependent(0.5)
    assert func2.is_dirac(2, {'d': 2})
    assert not func2.is_time_dependent(2, {'d': 2})
    assert func2.evaluate(0.5) == 0.5**2  # Collections sum interval together only
    assert func2.evaluate_dirac(0.5) == 1  # evaluation of dirac are done separately
    assert func2.times() == [0, 0.5, 1]


def test_composite_time_function_init():
    func0 = TimeIntervalFunction(value=lambda t, args: t**2, interval=[0, 1])
    func1 = TimeInstantFunction(value=1, instant='d', parameters={'d': 0.5})
    func2 = CompositeTimeFunction()
    assert not func2.has_instant
    assert not func2.has_interval
    func2.add(func0)
    assert func2.has_interval
    func2.add(func1)
    assert func2.evaluate(0.5) == 0.5 ** 2
    assert func2.times() == [0, 0.5, 1]
    assert func2.evaluate_dirac(0.5) == 1
    assert func2.has_instant
    assert func2.has_interval
