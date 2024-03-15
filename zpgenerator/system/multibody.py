from ..time import sum_tensor, tensor_dict, tensor_insert, id_flatten
from ..time.evaluate import EvaluatedQuadruple, EvaluatedDiracOperator
from .quantum import AQuantumSystem, SystemCollection
from .coupling import CouplingTerm, CouplingBase
from .control import ControlBase, CompositeControl
from .emitter import AQuantumEmitter
from typing import List, Union
from math import prod
from abc import abstractmethod
from qutip import Qobj


class AQuantumMultiBodyEmitter(AQuantumEmitter):
    """
    A quantum system composed of multiple subsystems
    """

    @property
    @abstractmethod
    def subsystems(self) -> dict:
        pass

    @property
    @abstractmethod
    def bodies(self) -> int:
        pass


class MultiBodyEmitterBase(AQuantumMultiBodyEmitter, SystemCollection):
    """
    A collection of independent quantum systems or emitters evaluated with a tensor product
    """

    def __init__(self,
                 subsystems: Union[AQuantumSystem, List[AQuantumSystem]] = None,
                 states: dict = None,
                 operators: dict = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        self._states = {} if states is None else states
        self._operators = {} if operators is None else operators
        self._subsystems = []
        super().__init__(systems=[], parameters=parameters, name=name, rule=sum_tensor,
                         types=types if types else [AQuantumSystem])
        if subsystems:
            self.add(subsystems)

        self._initial_state = None
        self._initial_time = None

    def _check_objects(self):
        self._check_keys()

    def _check_add(self, system, parameters: dict = None, name: str = None):
        self._subsystems.append(system)
        return system

    @property
    def states(self) -> dict:
        if not self._states or any(state.shape[0] != self.subdims for state in self._states.values()):
            if self._subsystems:
                self._states = self._subsystems[0].states
                for system in self._subsystems[1:]:
                    self._states = tensor_dict(self._states, system.states)
        return self._states

    @property
    def operators(self) -> dict:
        if not self._operators or any(op.shape[0] != self.subdims for op in self._operators.values()):
            self._operators = {}
            for i, system in enumerate(self._subsystems):
                self._operators.update({k: tensor_insert(v, i, self.subdims) for k, v in system.operators.items()})
        return self._operators

    @property
    def modes(self):
        return sum(system.modes for system in self._subsystems)

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Qobj):
        self._initial_state = state

    @property
    def initial_time(self):
        return self._initial_time

    @initial_time.setter
    def initial_time(self, time: Union[float, int]):
        self._initial_time = time

    @property
    def subsystems(self) -> dict:
        keys = self._make_unique_names(self._subsystems)
        return {keys[i]: system for i, system in enumerate(self._subsystems)}

    @property
    def bodies(self) -> int:
        return len(self._subsystems)

    @property
    def dim(self) -> int:
        return prod(system.dim for system in self._subsystems) if self._subsystems else None

    @property
    def subdims(self) -> list:
        dimset = [system.subdims for system in self._subsystems]
        return [dim for dims in dimset for dim in dims] if self._subsystems else None

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        parameters = self.set_parameters(parameters)
        return self._rule([system.evaluate_quadruple(t, parameters) for system in self._subsystems],
                          EvaluatedQuadruple())

    def gather_quadruples(self, t: float, parameters: dict = None) -> List[EvaluatedQuadruple]:
        return [self.evaluate_quadruple(t, parameters)]

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        parameters = self.set_parameters(parameters)
        return self._rule(id_flatten([op.evaluate_dirac(t, parameters) for op in self._subsystems]),
                          EvaluatedDiracOperator())


class MultiBodyEmitter(MultiBodyEmitterBase):
    """
    A collection of systems coupled together and with a tensor product
    """

    def __init__(self,
                 subsystems: Union[AQuantumSystem, List[AQuantumSystem]] = None,
                 coupling: CouplingBase = None,
                 control: ControlBase = None,
                 states: dict = None,
                 operators: dict = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        self.coupling = CouplingBase()
        self.control = CompositeControl()
        super().__init__(states=states, operators=operators, parameters=parameters, name=name, types=types)
        self._objects.append(self.coupling)
        self._objects.append(self.control)
        if subsystems:
            for system in subsystems:
                self.add(system)
        if coupling:
            self.add(coupling)
        if control:
            self.add(control)
        self._check_objects()

    def _check_objects(self):
        super()._check_objects()
        if self._objects and self.coupling.subdims:
            assert self.subdims == self.coupling.subdims, \
                "Coupling dimensions must match the dimensions of the coupled systems."

    def _check_add(self, system, parameters: dict = None, name: str = None):
        self._subsystems.append(system)
        return system

    def _add(self, system, parameters: dict = None, name: str = None):
        if isinstance(system, CouplingTerm) or isinstance(system, CouplingBase):
            self.coupling.add(system, parameters, name)
        elif isinstance(system, ControlBase):
            self.control.add(system, parameters, name)
        else:
            super()._add(system, parameters, name)
            self._extend_space(system.subdims)

    def _extend_space(self, subdims):
        self.coupling.pad_right(subdims)

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return super().evaluate_quadruple(t, parameters) + \
            self.coupling.evaluate_quadruple(t, self.set_parameters(parameters)) + \
            self.control.evaluate_quadruple(t, self.set_parameters(parameters))

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        parameters = self.set_parameters(parameters)
        return super().evaluate_dirac(t, parameters) + self.control.evaluate_dirac(t, self.set_parameters(parameters))
