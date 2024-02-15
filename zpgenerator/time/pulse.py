from .function import ATimeFunction, TimeInstantFunction
from .composite import CompositeTimeFunction
from .evaluate import Func
from typing import Union
from scipy.integrate import quad
from numpy import linspace, angle
from copy import copy
import matplotlib.pyplot as plt


class PulseBase(CompositeTimeFunction):
    """A base class for pulses with algorithms to visualize and manipulate functions"""

    def __init__(self,
                 functions: Union[ATimeFunction, list[ATimeFunction]] = None,
                 parameters: dict = None,
                 name: str = None,
                 cache=False):
        """
        :param functions: a TimeFunction or list of TimeFunctions describing the pulse shape.
        :param parameters: a dictionary of parameters needed to evaluate the pulse.
        :param name: a name for the object to distinguish parameters.
        """
        super().__init__(functions=functions, parameters=parameters, name=name)
        self._composition_rule = lambda amp, args: amp
        self.cache = cache


    def area(self, parameters: dict = None, lower_limit=None, upper_limit=None) -> float:
        parameters = self.set_parameters(parameters)
        interval = self.times(parameters)
        if len(interval) < 2:
            return float('inf')
        theta = quad(lambda t: abs(self.evaluate(t, parameters)),
                     interval[0] if lower_limit is None else lower_limit,
                     interval[-1] if upper_limit is None else upper_limit)[0]
        theta += sum(pulse.evaluate(parameters) for pulse in self.functions if isinstance(pulse, TimeInstantFunction))
        return theta

    def plot(self,
             resolution: int = 600,
             start: float = None,
             end: float = None,
             parameters: dict = None,
             scale: float = 1,
             label: str = None,
             xlabel: str = None,
             ylabel: str = None,
             function: callable = abs):
        parameters = self.set_parameters(parameters)
        interval = self.times(parameters)
        if len(interval) < 2:
            assert False, "No boundaries found, please specify."
        times = linspace((interval[0] if start is None else start) - 10 ** -12,
                         (interval[-1] if end is None else end) + 10 ** -12,
                         resolution)

        plt.plot(times, [function(self.evaluate(t, parameters)) * scale for t in times],
                 label=('Pulse' if self.name is None else self.name) if label is None else label)
        plt.ylabel(ylabel if ylabel else 'Shape (arb)')
        plt.xlabel(xlabel if xlabel else 'Time, $t$ ($T_1$)')
        plt.legend()
        return plt

    def plot_phase(self,
                   resolution: int = 600,
                   start: float = None,
                   end: float = None,
                   parameters: dict = None,
                   scale: float = 1,
                   label: str = None,
                   xlabel: str = None,
                   ylabel: str = None):
        return self.plot(resolution=resolution, start=start, end=end, parameters=parameters, scale=scale, label=label,
                         xlabel=xlabel, ylabel=ylabel, function=angle)

    def compose_with(self, function: callable, parameters: dict = None):
        current_rule = copy(self._composition_rule)
        self._composition_rule = lambda amp, args=None: function(current_rule(amp, args), args)
        if parameters:
            self._update_default_parameters(parameters)

    def evaluate_function(self, t: float, parameters: dict = None):
        set_parameters = self.set_parameters(parameters)
        func = self._rule([function.evaluate_function(t, set_parameters) for function in self._objects])
        if self._composition_rule:
            if isinstance(func, Func):
                func = func.compose_with(self._composition_rule, self.get_parameters(parameters))
            else:
                func = self._composition_rule(func, self.get_parameters(parameters))
        if isinstance(func, Func) and self.cache:
            func.cache = True
        return func

    def evaluate_dirac(self, t: float, parameters: dict = None):
        set_parameters = self.set_parameters(parameters)
        func = self._rule([function.evaluate_dirac(t, set_parameters) for function in self._objects])
        if self._composition_rule:
            if isinstance(func, Func):
                return func.compose_with(self._composition_rule, self.get_parameters(parameters))
            else:
                return self._composition_rule(func, self.get_parameters(parameters))
        else:
            return func


class Lifetime:
    """
    An object containing a list of times and values for the intensity of emission
    """

    def __init__(self, times: list, population: list):
        self.times = times
        self.population = [abs(pop) for pop in population]

    def plot(self, label: str = None, scale: float = 1):
        plt.plot(self.times, [p * scale for p in self.population], label='Lifetime' if label is None else label)
        plt.xlabel('Time, t ($T_1$)')
        plt.ylabel('Intensity (arb)')
        plt.legend()
        return plt
