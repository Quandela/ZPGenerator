from .parameters.parameterized_object import ParameterizedObject
from .domain import merge_intervals, merge_times
from .function import ATimeFunction, TimeFunction
from .evaluate import Func
from .parameters.collection import AParameterizedCollection, ParameterizedCollection
from abc import abstractmethod
from typing import Union, List, Callable


class ATimeFunctionCollection(ATimeFunction, AParameterizedCollection):
    """
    A collection of time functions or time operator functions, possibly of different types (instant, interval).
    """

    @property
    @abstractmethod
    def functions(self) -> dict:
        pass


class TimeFunctionCollection(ParameterizedCollection, ATimeFunctionCollection):
    """
    A collection of time functions or time operator functions, possibly of different types.
    """

    def __init__(self,
                 functions=None,
                 parameters: dict = None,
                 name: str = None,
                 rule: callable = sum,
                 types: list = None):
        self._rule = rule
        super().__init__(objects=functions, parameters=parameters, name=name,
                         types=[ATimeFunction, Callable, float, int, complex] if types is None else types)

    def _check_add(self, function, parameters: dict = None, name: str = None):
        function = super()._check_add(function, parameters, name)
        if isinstance(function, ATimeFunction):
            return function
        else:
            function = TimeFunction(value=function, parameters=parameters, name=name)
            if isinstance(function.value, ParameterizedObject):
                function.value.default_name = '_value'
            return function

    @property
    def functions(self) -> dict:
        keys = self._make_unique_names()
        return {keys[i]: func for i, func in enumerate(self._objects)}

    def support(self, t: float = None, parameters: dict = None) -> Union[bool, list]:
        parameters = self.set_parameters(parameters)
        if t is None:
            return merge_intervals([function.support(t, parameters) for function in self._objects])
        else:
            return any(function.support(t, parameters) for function in self._objects)

    def evaluate(self, t: float, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        return self._rule([function.evaluate(t, parameters) for function in self._objects], 0)

    def evaluate_dirac(self, t: float, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        return self._rule([function.evaluate_dirac(t, parameters) for function in self._objects], 0)

    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        parameters = self.set_parameters(parameters)
        return any(function.is_time_dependent(t, parameters) for function in self._objects)

    def is_dirac(self, t: float, parameters: dict = None) -> bool:
        parameters = self.set_parameters(parameters)
        return any(function.is_dirac(t, parameters) for function in self._objects)

    def times(self, parameters: dict = None) -> list:
        parameters = self.set_parameters(parameters)
        return merge_times([function.times(parameters) for function in self._objects])

    @property
    def has_instant(self):
        return any(function.has_instant for function in self._objects)

    @property
    def has_interval(self):
        return any(function.has_interval for function in self._objects)


class CompositeTimeFunction(TimeFunctionCollection):
    """
    A set of time functions that evaluates following summation, separates out any instant functions
    """

    def __init__(self,
                 functions: Union[ATimeFunction, List[ATimeFunction]] = None,
                 parameters: dict = None,
                 name: str = None,
                 rule: callable = sum):
        """
        :param functions: a list of time functions.
        :param parameters: a dictionary of default parameters to apply.
        :param name: a name for the object to distinguish parameters.
        """
        super().__init__(functions=functions, parameters=parameters, name=name, rule=rule)

    def _check_objects(self):
        self._check_keys()

    def evaluate_function(self, t: float, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        return self._rule([function.evaluate_function(t, parameters) for function in self._objects])

    def evaluate(self, t: float, parameters: dict = None):
        func = self.evaluate_function(t, parameters)
        return func(t) if isinstance(func, Func) else func

    def evaluate_dirac(self, t: float, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        return self._rule([function.evaluate_dirac(t, parameters) for function in self._objects])
