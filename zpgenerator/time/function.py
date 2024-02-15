from .parameters.parameterized_object import AParameterizedObject, ParameterizedObject
from .domain import TimeInterval, TimeInstant
from .evaluate.operator import Func
from abc import abstractmethod
from typing import Union
from inspect import signature


class ATimeFunction(AParameterizedObject):
    """
    A parameterized function of time on a domain, and that may have some other arguments.
    """

    # @abstractmethod
    # def evaluate_function(self, t: float, parameters: dict = None) -> Union[Func, any]:
    #     """
    #     Evaluates the function to a value or parameterized Func object describing the function in the interval
    #     following t that can be called to produce the numerical value at an exact time.
    #
    #     :param t: a real value.
    #     :param parameters: a dictionary of optional parameters used to modify the function.
    #     :return: the value of the function at the time t (if time-independent) or a Func object (if time dependent)
    #     """
    #     pass

    @abstractmethod
    def evaluate(self, t: float, parameters: dict = None):
        """
        Evaluates the function to a numerical value at time t.

        :param t: a real value.
        :param parameters: a dictionary of optional parameters used to modify the function.
        :return: the value of the function at the time t.
        """
        pass

    @abstractmethod
    def evaluate_dirac(self, t: float, parameters: dict = None):
        """
        Evaluates the function to a numerical value at time t if it is dirac at time t
        """
        pass

    @abstractmethod
    def support(self, t: float = None, parameters: dict = None) -> bool:
        """
        Whether the time is in the support of the defining function.

        :param t: a real value.
        :param parameters: a dictionary of optional parameters used to modify the function.
        :return: a boolean
        """
        pass

    # A list of finite times in non-decreasing order defining the domain of the function
    @abstractmethod
    def times(self, parameters: dict = None) -> list:
        """
        A list of finite times in non-decreasing order describing important stopping points inherited from its domain

        :param parameters: a dictionary of parameters used to evaluate the times.
        :return: a list of times.
        """
        pass

    @abstractmethod
    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        """
        Whether the function is time dependent at time t (derivative is nonzero in the region immediately after t)

        :param t: a real value.
        :param parameters: a dictionary of optional arguments used to modify the function.
        :return: a boolean
        """
        pass

    @abstractmethod
    def is_dirac(self, t: float, parameters: dict = None) -> bool:
        """
        Whether the function is a dirac delta at time t (support is a point at t).

        :param t: a real value.
        :param parameters: a dictionary of optional parameters used to modify the function.
        :return: a boolean
        """
        pass

    @property
    @abstractmethod
    def has_interval(self):
        pass

    @property
    @abstractmethod
    def has_instant(self):
        pass


class TimeFunction(ParameterizedObject, ATimeFunction):
    """
    A parameterized function of time (possibly constant) defined on the entire real line.
    """

    def __init__(self, value, parameters: dict = None, name: str = None):
        """
        :param value: a function f(t, parameters) or f(parameters) that returns a result, or a constant value.
        :param parameters: a dictionary of parameters needed to evaluate the value function.
        :param name: a name for the object to distinguish parameters.
        """
        super().__init__(parameters=parameters, name=name)
        self.value = value
        self.is_callback = callable(value)
        self._is_time_dependent = self.is_callback and 't' in signature(value).parameters

    def evaluate_function(self, t: float, parameters: dict = None) -> Union[Func, any]:
        parameters = self.get_parameters(parameters)
        if self.is_callback:
            if self._is_time_dependent:
                return Func(self.value, args=parameters)
            else:
                return self.value(parameters)
        else:
            return self.value

    def support(self, t: float = None, parameters: dict = None) -> bool:
        return True

    def evaluate(self, t: float, parameters: dict = None):
        func = self.evaluate_function(t, parameters)
        return func(t) if isinstance(func, Func) else func

    def evaluate_dirac(self, t: float, parameters: dict = None):
        return 0

    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        return self._is_time_dependent

    def is_dirac(self, t: float, parameters: dict = None) -> bool:
        return False

    def times(self, parameters: dict = None) -> list:
        return []

    @property
    def has_interval(self):
        return True

    @property
    def has_instant(self):
        return False


class TimeIntervalFunction(ParameterizedObject, ATimeFunction):
    """
    A function of time that may depend on time in an interval of time but is constant elsewhere.
    """

    def __init__(self,
                 value,
                 interval: Union[list, callable, TimeInterval] = None,
                 default=None,
                 freeze: bool = False,
                 parameters: dict = None,
                 name: str = None):
        """
        :param value: a function f(t, parameters) or f(parameters) that returns a value when inside the interval.
        :param interval: a list [begin, end], a function that returns a list [begin end], or a TimeInterval object.
        :param default: the value given by the object when evaluated outside the interval.
        :param freeze: True adds to the default with the value given by the function at the closest interval endpoint.
        :param parameters: a dictionary of default parameters needed to evaluate any callable inputs.
        :param name: a name for the object to distinguish parameters.
        """
        self.value = value if isinstance(value, TimeFunction) else TimeFunction(value=value, parameters=parameters)
        self.value.default_name = '_value'

        self.interval = interval if isinstance(interval, TimeInterval) \
            else TimeInterval(interval=interval, parameters=parameters)
        self.interval.default_name = '_interval'

        self.default = default if isinstance(default, TimeFunction) \
            else TimeFunction(value=0 if default is None else default, parameters=parameters)
        self.default.default_name = '_default'

        super().__init__(parameters=parameters, name=name, children=[self.value, self.interval, self.default])

        self.freeze = freeze

    def support(self, t: float = None, parameters: dict = None) -> Union[bool, list]:
        parameters = self.set_parameters(parameters)
        return self._support(t, parameters)

    def _support(self, t: float = None, parameters: dict = None) -> Union[bool, list]:
        if t is None:
            return self.interval.evaluate(parameters)
        else:
            interval = self.interval.evaluate(parameters)
            return interval[0] <= t < interval[-1]

    def evaluate(self, t: float, parameters: dict = None):
        func = self.evaluate_function(t, parameters)
        return func(t) if isinstance(func, Func) else func

    def evaluate_function(self, t: float, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        if self._support(t, parameters):
            return self.value.evaluate_function(t, parameters)
        else:
            if self.freeze:
                interval = self.interval.evaluate(parameters)
                if t < interval[0]:
                    freeze_value = self.value.evaluate(interval[0], parameters)
                else:
                    freeze_value = self.value.evaluate(interval[-1], parameters)
            else:
                freeze_value = 0
            return self.default.evaluate(t, parameters) + freeze_value

    def evaluate_dirac(self, t: float, parameters: dict = None):
        return 0

    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        parameters = self.set_parameters(parameters)
        return self.support(t, parameters) if self.value.is_time_dependent(t, parameters) else False

    def is_dirac(self, t: float, parameters: dict = None) -> bool:
        return False

    def times(self, parameters: dict = None) -> list:
        parameters = self.set_parameters(parameters)
        return self.interval.times(parameters)

    @property
    def has_interval(self):
        return True

    @property
    def has_instant(self):
        return False


class TimeInstantFunction(ParameterizedObject, ATimeFunction):
    """A Dirac delta function."""

    def __init__(self,
                 value,
                 instant: Union[list, callable, TimeInstant],
                 default=None,
                 parameters: dict = None,
                 name: str = None):
        """
        :param value: a function f(parameters) that returns a result, or a constant value (not callable).
        :param instant: a constant list or a function f(parameters) that returns a list of one real value.
        :param default: the value given by the object when evaluated outside the interval.
        :param parameters: a dictionary of default parameters needed to evaluate any callable inputs.
        :param name: a name for the object to distinguish parameters.
        """
        self.value = value if isinstance(value, TimeFunction) else \
            TimeFunction(value=value, parameters=parameters)
        self.value.default_name = '_value'

        self.instant = instant if isinstance(instant, TimeInstant) else \
            TimeInstant(instant=instant, parameters=parameters)
        self.instant.default_name = '_value'

        self.default = default if isinstance(default, TimeFunction) \
            else TimeFunction(value=0 if default is None else default, parameters=parameters)
        self.default.default_name = '_default'

        super().__init__(parameters=parameters, name=name, children=[self.value, self.instant, self.default])

    def support(self, t: float = None, parameters: dict = None) -> Union[bool, list]:
        parameters = self.set_parameters(parameters)
        if t is None:
            return self.instant.evaluate(parameters)
        else:
            return t in self.instant.evaluate(parameters)

    def evaluate(self, t: float, parameters: dict = None):
        return 0

    def evaluate_function(self, t: float, parameters: dict) -> Union[any, Func]:
        return 0

    def evaluate_dirac(self, t: float, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        if self.support(t, parameters):
            return self.value.evaluate(t, parameters)
        else:
            return self.default.evaluate(t, parameters)

    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        return False


    def is_dirac(self, t: float, parameters: dict = None) -> bool:
        parameters = self.set_parameters(parameters)
        return self.support(t, parameters)


    def times(self, parameters: dict = None) -> list:
        parameters = self.set_parameters(parameters)
        return self.instant.times(parameters)

    @property
    def has_interval(self):
        return False

    @property
    def has_instant(self):
        return True
