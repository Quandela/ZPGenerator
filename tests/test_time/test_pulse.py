from zpgenerator.time.pulse import *
from zpgenerator.time.function import *
from pytest import raises


def test_pulse_base_inf():
    pulse = PulseBase()
    func = TimeIntervalFunction(value=1)
    pulse.add(func)
    assert pulse.evaluate(0) == 1
    assert pulse.area() == float('inf')
    with raises(AssertionError):
        pulse.plot()


def test_pulse_base_bounds():
    pulse = PulseBase()
    func = TimeIntervalFunction(value=1, interval=[-1, 1])
    pulse.add(func)
    assert pulse.evaluate(0) == 1
    assert pulse.evaluate(1) == 0
    assert pulse.area() == 2
    assert pulse.area(lower_limit=0) == 1
    assert pulse.plot()


def test_pulse_base_sum():
    pulse = PulseBase()
    pulse.add(TimeIntervalFunction(value=1, interval=[-1, 1]))
    pulse.add(TimeIntervalFunction(value=lambda t, args: t**2, interval=[-1, 3]))
    assert pulse.evaluate(-1) == 2
    assert pulse.evaluate(0) == 1
    assert pulse.evaluate(1) == 1
    assert pulse.evaluate(2) == 4
    assert pulse.area() == 2 + 28/3
    assert pulse.area(lower_limit=0) == 10
    assert pulse.plot()


def test_pulse_compose():
    pulse = PulseBase()
    pulse.add(TimeIntervalFunction(value=lambda t, args: args['a'] * t**2, interval=[-1, 3], parameters={'a': 2}))
    pulse.compose_with(lambda amp, args: 1 / amp if amp else 0)
    assert pulse.evaluate(-1) == 1 / 2
    assert pulse.evaluate(2) == 1 / 8
    assert pulse.evaluate(4) == 0

