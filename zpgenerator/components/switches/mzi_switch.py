from ..circuits.base_circuit import CircuitComponent
from ...system import CompositeScatterer
from ...time import TimeInterval, TimeOperator, TimeIntervalFunction
from typing import Union
from qutip import Qobj
from numpy import exp, pi


class MZISwitch(CircuitComponent):
    """ A time-dynamic linear-optical component that opens one or more modes during a window of time."""
    def __init__(self, switch_function: callable, interval: Union[list, callable, TimeInterval],
                 default: float = None, freeze: bool = True, parameters: dict = None, name: str = None):
        interval = interval if isinstance(interval, TimeInterval) else TimeInterval(interval, parameters=parameters)
        func = TimeIntervalFunction(value=lambda t, args: exp(1.j * switch_function(t, args) * pi),
                                    interval=interval, parameters=parameters, default=default, freeze=freeze)
        operator = TimeOperator(Qobj([[1, 1], [1, 1]]) / 2, func)
        element = CompositeScatterer(operator)
        element.add(TimeOperator(Qobj([[1, -1], [-1, 1]]) / 2))
        super().__init__(element, name=name)
        self.default_name = '_Switch'
