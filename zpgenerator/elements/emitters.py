from .nonlinear import TrionEmitter, ExcitonEmitter, BiexcitonEmitter, TwoLevelEmitter, CavityEmitter, \
    ShapedLaserEmitter, PurcellEmitter
from ..system import EmitterBase
from ..time import PulseBase, Lifetime
from typing import Union


class Emitter(EmitterBase):
    """
    A quantum emitter factory, produces systems where the LindbladVector represents dipole operators
    """

    @classmethod
    def two_level(cls, parameters: dict = None, name: str = None, modes: int = 1):
        return TwoLevelEmitter(parameters=parameters, name=name, modes=modes)

    @classmethod
    def exciton(cls, parameters: dict = None, name: str = None):
        return ExcitonEmitter(parameters=parameters, name=name)

    @classmethod
    def biexciton(cls, parameters: dict = None, name: str = None):
        return BiexcitonEmitter(parameters=parameters, name=name)

    @classmethod
    def trion(cls, charge: str = 'negative', parameters: dict = None, name: str = None):
        return TrionEmitter(charge=charge, parameters=parameters, name=name)

    @classmethod
    def cavity(cls, truncation: int = 2, parameters: dict = None, name: str = None, modes: int = 1):
        return CavityEmitter(truncation=truncation, modes=modes, parameters=parameters, name=name)

    @classmethod
    def oscillator(cls, shape: Union[PulseBase, Lifetime], resolution: int = 600, truncation: int = 2,
                   parameters: dict = None, name: str = None):
        return ShapedLaserEmitter(shape=shape, resolution=resolution, truncation=truncation,
                                  parameters=parameters, name=name)

    @classmethod
    def purcell(cls, purcell_factor: float = None, regime: float = None, timescale: float = None,
                parameters: dict = None, name: str = None):
        return PurcellEmitter(purcell_factor=purcell_factor, regime=regime, timescale=timescale,
                              parameters=parameters, name=name)
