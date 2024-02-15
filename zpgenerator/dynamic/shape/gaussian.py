from .functions import gaussian, sigma_window
from ...time import TimeFunction, TimeInterval, TimeIntervalFunction, PulseBase
from ...time.parameters import parinit
from numpy import pi


class GaussianPulse(PulseBase):
    """
    A Gaussian pulse function defined on a restricted interval.
    """
    def __init__(self, parameters: dict = None, name: str = None):
        """
        :param parameters: the default parameters 'area', 'width', 'delay', 'detuning', 'window' of the gaussian pulse.
        :param name: the name of the pulse that modifies the parameter names to distinguish it from other pulses.
        """
        value_parameters = parinit({'width': 0.1, 'area': pi, 'delay': 0, 'detuning': 0, 'phase': 0}, parameters)

        interval_parameters = parinit({'width': 0.1, 'delay': 0, 'window': 6}, parameters)

        function = TimeIntervalFunction(value=TimeFunction(value=gaussian, parameters=value_parameters),
                                        interval=TimeInterval(interval=sigma_window, parameters=interval_parameters))

        super().__init__(functions=function, parameters=value_parameters | interval_parameters, name=name)
#
#
# class GaussianSpectralPulse(PulseBase):
#     """
#     A Gaussian pulse in frequency.
#     """
#     def __init__(self, parameters: dict = None, name: str = None):
#         """
#         :param parameters: the default parameters 'area', 'width', 'delay', 'detuning', 'window' of the gaussian pulse.
#         :param name: the name of the pulse that modifies the parameter names to distinguish it from other pulses.
#         """
#         value_parameters = parinit({'width': 0.1, 'area': pi, 'delay': 0, 'detuning': 0}, parameters)
#
#         interval_parameters = parinit({'width': 0.1, 'delay': 0, 'window': 6}, parameters)
#
#         function = TimeIntervalFunction(value=TimeFunction(value=gaussian, parameters=value_parameters),
#                                         interval=TimeInterval(interval=sigma_window, parameters=interval_parameters))
#
#         super().__init__(functions=function, parameters=value_parameters | interval_parameters, name=name)
