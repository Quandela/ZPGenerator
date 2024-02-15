from ...system import EmitterBase, LindbladVector
from ...time import TimeOperator, TimeIntervalFunction, PulseBase, Lifetime
from .cavity import ShapedCavitySystem
from ...time.parameters import parinit
from numpy import sqrt
from qutip import qeye
from typing import Union


class ShapedLaserSystem(ShapedCavitySystem):
    """
    Shaped cavity emission with quantum fluctuations and classical offset to make a laser system
    """

    def __init__(self,
                 shape: Union[PulseBase, Lifetime],
                 resolution: int = 600,
                 truncation: int = 2,
                 parameters: dict = None,
                 name: str = None):
        super().__init__(shape=shape, resolution=resolution, truncation=truncation, parameters=parameters, name=name)

        offset = TimeIntervalFunction(value=lambda t, args: args['amplitude'] * sqrt(self.shape_function(t, args)),
                                      interval=self.interval,
                                      parameters=parinit({'amplitude': 0, 'decay': 1, 'delay': 0}, parameters))

        self.environment.operator_list[1].add(TimeOperator(operator=qeye(truncation), functions=offset))


class ShapedLaserEmitter(EmitterBase):
    """
    Quantum fluctuations and classical offset for a laser with emission transition
    """

    def __init__(self,
                 shape: Union[PulseBase, Lifetime],
                 resolution: int = 600,
                 truncation: int = 2,
                 parameters: dict = None,
                 name: str = None):
        system = ShapedLaserSystem(shape=shape, resolution=resolution, truncation=truncation,
                                   parameters=parameters, name=name)

        transitions = LindbladVector(operators=system.environment.operator_list[1], parameters=parameters)

        super().__init__()
        self.set_system(system=system, transitions=transitions)
