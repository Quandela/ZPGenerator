from ..time import ATimeOperator, Operator, TimeVectorOperator
from .scatterer import AElement
from .natural import AQuantumSystem, HamiltonianBase, EnvironmentBase
from .control import ChannelBase, ControlBase, CompositeControl, ControlledSystem
from ..time.evaluate.quadruple import EvaluatedQuadruple
from typing import Union, List
from qutip import Qobj
from abc import abstractmethod


# maybe could be made a subclass of EnvironmentBase?
class LindbladVector(AQuantumSystem, TimeVectorOperator):
    """
    A vector of mode-coupling Lindblad collapse operators that may have time-dependent amplitudes.
    """

    def __init__(self,
                 operators: Union[List[ATimeOperator], List[Qobj], List[Operator]] = None,
                 parameters: dict = None,
                 name: str = None):
        super().__init__(operators=operators, parameters=parameters, name=name)

    # The number of collapse operators
    @property
    def modes(self):
        modes = 0
        for op in self._objects:
            modes += (op.modes if isinstance(op, LindbladVector) else 1)  # allows for nested LindbladVectors
        return modes

    def _check_objects(self):
        super()._check_objects()
        assert all(not op.is_super for op in self._objects), "Cannot add superoperators."
        assert all(not op.has_instant for op in self._objects), "Cannot add instant operators."

    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None):
        return self.is_time_dependent(t, parameters)

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return EvaluatedQuadruple(transitions=super().partial_evaluate(t, parameters))


class AQuantumEmitter(AQuantumSystem, AElement):

    @property
    @abstractmethod
    def initial_state(self):
        pass

    @property
    @abstractmethod
    def initial_time(self):
        pass

    @property
    @abstractmethod
    def modes(self) -> int:
        pass

    @property
    def is_emitter(self) -> bool:
        return True


class EmitterBase(AQuantumEmitter, ControlledSystem):
    """
    A controlled quantum system with a list of transition operators specifying coupling modes to emit light
    """

    def __init__(self,
                 hamiltonian: Union[HamiltonianBase, Qobj, Operator] = None,
                 environment: Union[EnvironmentBase, List[Qobj], List[Operator]] = None,
                 control: Union[CompositeControl, ControlBase, HamiltonianBase, EnvironmentBase, ChannelBase] = None,
                 transitions: Union[LindbladVector, List[Qobj], List[ATimeOperator]] = None,
                 states: dict = None,
                 operators: dict = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):

        self.transitions = LindbladVector(transitions) if not isinstance(transitions, LindbladVector) else \
            LindbladVector() if transitions is None else transitions
        self.transitions.default_name = '_transitions'

        super().__init__(hamiltonian=hamiltonian, environment=environment, control=control,
                         states=states, operators=operators, parameters=parameters, name=name,
                         types=[HamiltonianBase, EnvironmentBase, ChannelBase, ControlBase, LindbladVector]
                         if types is None else types)
        self._objects.append(self.transitions)
        self._check_objects()

        self._initial_time = None
        self._initial_state = None

        self.system = None

    def _check_objects(self):
        super()._check_objects()
        if self.transitions.operator_list:
            assert self.environment.operator_list, "Transitions cannot occur without an environment"
            assert self.subdims == self.transitions.subdims, \
                "Transition operator dimensions must match the dimensions of the system."

    @property
    def modes(self):
        return self.transitions.modes

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

    def set_system(self,
                   system: AQuantumSystem,
                   transitions: Union[LindbladVector, List[Qobj], List[ATimeOperator]] = None):
        EmitterBase.__init__(self,
                             hamiltonian=system.hamiltonian if hasattr(system, 'hamiltonian') else None,
                             environment=system.environment if hasattr(system, 'environment') else None,
                             control=system.control if hasattr(system, 'control') else None,
                             transitions=transitions,
                             states=system.states if hasattr(system, 'states') else None,
                             operators=system.operators if hasattr(system, 'operators') else None,
                             parameters=system.local_default_parameters,
                             name=system.name)
        self.system = system

    def _add(self, system, parameters: dict = None, name: str = None):
        super()._add(system, parameters, name)
        if isinstance(system, LindbladVector):
            self.transitions.add(system, parameters, name)

    def gather_quadruples(self, t: float, parameters: dict = None) -> List[EvaluatedQuadruple]:
        return [self.evaluate_quadruple(t, parameters)]
