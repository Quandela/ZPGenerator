from ..circuits.base_circuit import CircuitComponent
from ...time import Operator
from ...time.parameters import parinit
from ...system import ScattererBase
from typing import Union
from qutip import qeye
from numpy import sqrt


class UniformLoss(CircuitComponent):
    """ A component that reduces the transmission efficiency of a mode """

    def __init__(self, efficiency: Union[float, int] = 1, modes: int = 1, parameters: dict = None, name: str = None):
        element = ScattererBase(Operator(lambda args: sqrt(args['efficiency']) * qeye(modes),
                                         parameters=parinit({'efficiency': efficiency}, parameters)))
        super().__init__(elements=element, name=name)
        self.default_name = '_Loss'
