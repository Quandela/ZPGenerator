from zpgenerator.time.domain import *
from pytest import raises
from zpgenerator.time.parameters import Parameters

d = Parameters.DELIMITER

def test_time_interval_init_empty():
    window = TimeInterval()
    assert window.interval == [float('-inf'), float('inf')]
    assert window._default_parameters == {}
    assert window.default_name == '_TimeInterval'
    assert window.name is None
    assert not window.is_callback
    assert window.evaluate() == [float('-inf'), float('inf')]
    assert window.times() == []


def test_time_interval_string():
    window = TimeInterval(interval=['begin', 1], parameters={'begin': 0})
    assert window._default_parameters == {'begin': 0}
    assert window.is_callback
    assert window.evaluate() == [0, 1]
    assert window.times() == [0, 1]
    assert window.evaluate({'begin': -1}) == [-1, 1]
    assert window.times({'begin': -1}) == [-1, 1]

    window = TimeInterval(interval=[0, 'end'], parameters={'end': 1})
    assert window._default_parameters == {'end': 1}
    assert window.is_callback
    assert window.evaluate() == [0, 1]
    assert window.times() == [0, 1]
    assert window.evaluate({'end': 2}) == [0, 2]
    assert window.times({'end': 2}) == [0, 2]
    assert window.evaluate({'end':-1}) == [0, -1]

    with raises(AssertionError):
        TimeInterval(interval=['begin', 'end'], parameters={'begin': 2, 'end': 1})

    window = TimeInterval(interval=['begin', 'end'], parameters={'begin': 0, 'end': 1})
    assert window._default_parameters == {'begin': 0, 'end': 1}
    assert window.is_callback
    assert window.evaluate() == [0, 1]
    assert window.times() == [0, 1]
    assert window.evaluate({'begin': -1, 'end': 2}) == [-1, 2]
    assert window.times({'begin': -1, 'end': 2}) == [-1, 2]


def test_time_interval_callable():
    window = TimeInterval(interval=lambda arg: [arg['delay'] - arg['window']/2, arg['delay'] + arg['window']/2],
                          parameters={'delay': 0, 'window': 1})
    assert window.evaluate() == [-1/2, 1/2]
    assert window.evaluate({'delay': 1/2}) == [0, 1]
    assert window.evaluate({'delay': 1/2, 'window': 3}) == [-1, 2]


def test_time_instant_string():
    obj = TimeInstant(instant=0)
    assert not obj.is_callback
    assert obj.instant == [0]
    assert obj.evaluate() == [0]
    assert obj.times() == [0]


def test_time_instant_str():
    obj = TimeInstant(instant='switch', parameters={'switch': 0})
    assert obj.is_callback
    assert obj.evaluate() == [0]
    assert obj.times() == [0]
    assert obj.evaluate({'switch': 1.5}) == [1.5]


def test_time_instant_parameter_names():
    obj = TimeInstant(instant='time', parameters={'time': 0}, name='switch')
    assert obj.is_callback
    assert obj.evaluate() == [0]
    assert obj.times() == [0]
    assert obj.evaluate({'switch' + d + 'time': 1.5}) == [1.5]
    assert obj.name == 'switch'
    assert obj.parameters == ['time']
    assert obj.default_parameters == {'time': 0}
    assert obj.parameter_tree({'time': 1}) == {'time': 1}
    assert obj.parameter_tree({'switch' + d + 'time': 2}) == {'time': 2}


def test_merge_intervals():
    assert merge_intervals([[1, 4], [3, 6], [7, 8]]) == [[1, 6], [7, 8]]
    assert merge_intervals([[1, 4], [7, 8], [3, 6]]) == [[1, 6], [7, 8]]


def test_merge_times():
    assert merge_times([[1, 2, 5, 8], [2, 4, 9]]) == [1, 2, 4, 5, 8, 9]
    assert merge_times([[2, 4, 9], [2, 1, 5, 8]]) == [1, 2, 4, 5, 8, 9]
