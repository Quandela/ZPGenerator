from .functions import square_amplitude, square_amplitude_detuned, window
from ...time import TimeFunction, TimeInterval, TimeIntervalFunction, PulseBase
from ...time.parameters import parinit
from numpy import pi


class SquarePulse(PulseBase):
    """
    A Square pulse function.
    """
    def __init__(self, parameters: dict = None, name: str = None):
        """
        :param parameters: the default parameters 'area', 'width', 'delay' of the pulse.
        :param name: the name of the pulse that modifies the parameter names to distinguish it from other pulses.
        """
        parameters = parameters if parameters else {}
        interval_parameters = parinit({'width': 0.1, 'delay': 0}, parameters)
        if 'detuning' in parameters.keys():
            value_parameters = parinit({'width': 0.1, 'area': pi, 'detuning': 0, 'phase': 0}, parameters)
            function = TimeIntervalFunction(value=TimeFunction(value=square_amplitude_detuned,
                                                               parameters=value_parameters),
                                            interval=TimeInterval(interval=window, parameters=interval_parameters))
        else:
            value_parameters = parinit({'width': 0.1, 'area': pi, 'phase': 0}, parameters)
            function = TimeIntervalFunction(value=TimeFunction(value=square_amplitude, parameters=value_parameters),
                                            interval=TimeInterval(interval=window, parameters=interval_parameters))

        super().__init__(functions=function, parameters=value_parameters | interval_parameters, name=name)
