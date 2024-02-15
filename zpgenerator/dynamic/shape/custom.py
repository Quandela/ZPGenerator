from ...time import TimeFunction, TimeInterval, TimeIntervalFunction, PulseBase
from typing import Union
from scipy.integrate import quad


class CustomPulse(PulseBase):
    """
    A custom pulse shape function defined on a restricted interval.
    """

    def __init__(self, shape: callable, gate: Union[list, callable], auto_normalise: bool = False, norm: float = 1,
                 parameters: dict = None, name: str = None):
        """
        :param shape: a function of the form f(t, args) that determines the shape of the pulse.
        :param gate: a list [begin, end] that determines when to evaluate the shape function.
        :param auto_normalise: set True to automatically re-define the shape to integrate over the gate to 1.
        :param norm: the value shape should integrate to when provided with the default parameters.
        :param parameters: the default parameters needed to evaluate the shape and gate.
        :param name: the name of the pulse that modifies the parameter names to distinguish it from other pulses.
        """

        gate = TimeInterval(interval=gate, parameters=parameters)

        if auto_normalise:
            interval = gate.evaluate(parameters=parameters)
            norm0 = quad(lambda t: abs(shape(t, parameters)), a=interval[0], b=interval[-1])[0]
            value = lambda t, args: norm * shape(t, args) / norm0
        else:
            value = shape

        function = TimeIntervalFunction(value=TimeFunction(value=value, parameters=parameters), interval=gate)

        super().__init__(functions=function, parameters=parameters, name=name)
