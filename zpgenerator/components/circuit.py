from .losses.uniform import UniformLoss
from .switches import *
from ..elements.linear import *
from ..network import Component
from ..time import TimeInterval
from ..time.parameters import parinit
from ..system import ScattererBase
from typing import Union
from qutip import Qobj
from typing import List


class Circuit(Component):
    """ A linear-optical circuit factory """
    @classmethod
    def haar_random(cls, modes: int, name: str = None):
        return Component(ScattererBase.haar_random(modes), name=name)

    @classmethod
    def custom(cls, matrix, parameters: dict = None, name: str = None):
        return Component(ScattererBase(matrices=matrix if isinstance(matrix, Qobj) or callable(matrix) else Qobj(matrix),
                                       parameters=parameters), name=name)

    @classmethod
    def loss(cls, efficiency: float = 1, modes: int = 1, parameters: dict = None, name: str = None):
        return UniformLoss(efficiency=efficiency, modes=modes, parameters=parameters, name=name)

    @classmethod
    def bs(cls, angle: float = None, parameters: dict = None, name: str = None):
        parameters = parinit({'angle': angle}, parameters) if angle is not None else parameters
        component = Component(BeamSplitter(parameters=parameters), name=name)
        component.default_name = '_BS'
        return component

    @classmethod
    def ps(cls, phase: float = None, parameters: dict = None, name: str = None):
        parameters = parinit({'phase': phase}, parameters) if phase is not None else None
        component = Component(PhaseShifter(parameters=parameters), name=name)
        component.default_name = '_PS'
        return component

    @classmethod
    def perm(self, perm: List[int] = None):
        return Component(Permutation(perm if perm else [1, 0]))

    @classmethod
    def mzi(cls):
        mzi = cls.bs(name='BS 1')
        mzi.add(0, cls.loss(name='upper loss'))
        mzi.add(1, cls.loss(name='lower loss'))
        mzi.add(0, cls.ps(name='upper 2'))
        mzi.add(1, cls.ps(name='lower phase 2'))
        mzi.add(0, cls.bs(name='BS 2'))
        mzi.default_name = '_MZI'
        return mzi

    @classmethod
    def gate(cls, interval: Union[list, TimeInterval], modes: int = 1, parameters: dict = None, name: str = None):
        return Gate(interval=interval, modes=modes, parameters=parameters, name=name)

    @classmethod
    def switch(cls, function: callable, interval: Union[list, TimeInterval],
               default: float = None, freeze: bool = True, parameters: dict = None, name: str = None):
        return MZISwitch(switch_function=function, interval=interval, default=default, freeze=freeze,
                         parameters=parameters, name=name)

    @classmethod
    def from_perceval(cls, circuit):
        assert hasattr(circuit, 'compute_unitary'), "Circuit object must have a 'compute_unitary' method. "
        component = Component(ScattererBase(Qobj(circuit.compute_unitary())))
        component.default_name = '_PCVL'
        return component
