from .functions import amplitude, amplitude_detuned
from ...time import TimeFunction, PulseBase
from ...time.parameters import parinit


class CWPulse(PulseBase):
    """
    A continuous-wave laser function.
    """
    def __init__(self, parameters: dict = None, name: str = None):
        """
        :param parameters: the default parameters 'amplitude' and 'detuning' of the laser.
        :param name: the name of the pulse that modifies the parameter names to distinguish it from other pulses.
        """
        parameters = parameters if parameters else {}
        if 'detuning' in parameters.keys():
            parameters = parinit({'amplitude': 1., 'detuning': 0, 'phase': 0}, parameters)
            function = TimeFunction(value=amplitude_detuned, parameters=parameters)
        else:
            parameters = parinit({'amplitude': 1., 'phase': 0}, parameters)
            function = TimeFunction(value=amplitude, parameters=parameters)

        super().__init__(functions=function, parameters=parameters, name=name)
