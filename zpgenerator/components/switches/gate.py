from ..circuits.base_circuit import CircuitComponent
from ...system import ScattererBase
from ...time import TimeInterval, TimeOperator, TimeIntervalFunction
from typing import Union
from qutip import qeye


class Gate(CircuitComponent):
    """ A time-dynamic linear-optical component that opens one or more modes during a window of time."""
    def __init__(self, interval: Union[list, callable, TimeInterval], modes: int = 1,
                 parameters: dict = None, name: str = None):
        interval = interval if isinstance(interval, TimeInterval) else TimeInterval(interval, parameters=parameters)
        operator = TimeOperator(qeye(modes), TimeIntervalFunction(1, interval, default=0))
        element = ScattererBase(operator)
        super().__init__(element, name=name)
        self.default_name = '_Gate'
