from ...elements import ShapedLaserEmitter
from ...time import TimeInterval, Lifetime
from ...dynamic.pulse import Pulse
from .base_source import GatedSourceComponent
from typing import Union


class ShapedLaserSource(GatedSourceComponent):
    """
    A source of a shaped local oscillator pulse
    """

    def __init__(self,
                 shape: Union[Pulse, Lifetime, callable] = None,
                 resolution: int = 1000,
                 truncation: int = 2,
                 gate: Union[TimeInterval, list] = None,
                 efficiency: float = 1,
                 parameters: dict = None,
                 name: str = None):

        lo = ShapedLaserEmitter(shape=shape, resolution=resolution, truncation=truncation,
                                parameters=parameters, name=name)

        lo.initial_state = lo.states['|0>']

        super().__init__(emitter=lo, gate=gate, efficiency=efficiency, name=name, parameters=parameters)
        self.default_name = '_Laser'
