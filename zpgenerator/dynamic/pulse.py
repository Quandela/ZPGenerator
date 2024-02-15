from .shape import *
from ..time import PulseBase
from typing import Union


class Pulse(PulseBase):
    """A pulse factory"""

    def __floordiv__(self, other):
        self.add(other)
        return self

    @classmethod
    def gaussian(cls, parameters: dict = None, name: str = None):
        return GaussianPulse(parameters=parameters, name=name)

    @classmethod
    def square(cls, parameters: dict = None, name: str = None):
        return SquarePulse(parameters=parameters, name=name)

    @classmethod
    def dirac(cls, parameters: dict = None, name: str = None):
        return DiracPulse(parameters=parameters, name=name)

    @classmethod
    def custom(cls, shape: callable, gate: Union[list, callable], auto_normalise: bool = False, norm: float = 1,
               parameters: dict = None, name: str = None):
        def custom_pulse_shape(t, args):
            try:
                return shape(t, args)
            except ValueError:
                return 0

        return CustomPulse(shape=custom_pulse_shape, gate=gate, auto_normalise=auto_normalise, norm=norm,
                           parameters=parameters, name=name)

    @classmethod
    def cw(cls, parameters: dict, name: str = None):
        return CWPulse(parameters=parameters, name=name)
