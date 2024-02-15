from .functions import delay, area
from ...time import TimeFunction, TimeInstant, TimeInstantFunction, PulseBase
from ...time.parameters import parinit
from numpy import pi


class DiracPulse(PulseBase):
    """
    A Dirac delta pulse function.
    """
    def __init__(self, parameters: dict = None, name: str = None):
        """
        :param parameters: the default parameters 'area', 'width', 'delay' and 'phase' of the pulse.
        :param name: the name of the pulse that modifies the parameter names to distinguish it from other pulses.
        """
        value_parameters = parinit({'area': pi, 'phase': 0}, parameters)

        instant_parameters = parinit({'delay': 0}, parameters)

        function = TimeInstantFunction(value=TimeFunction(value=area, parameters=value_parameters),
                                       instant=TimeInstant(instant=delay, parameters=instant_parameters))

        super().__init__(functions=function, parameters=value_parameters | instant_parameters, name=name)
