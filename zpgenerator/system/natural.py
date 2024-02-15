from ..time import OperatorInputList, CompositeTimeOperator, TimeVectorOperator, TimeFunctionCollection
from ..time.evaluate.quadruple import EvaluatedQuadruple
from .quantum import AQuantumSystem, SystemCollection
from typing import Union


class HamiltonianBase(AQuantumSystem, CompositeTimeOperator):
    """
    A HamiltonianBase operator that may be dependent on time.
    """

    def __init__(self,
                 operators: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        super().__init__(operators=operators, parameters=parameters, name=name, types=types)

    def _check_objects(self):
        self._check_keys()
        assert all(op.subdims == self._objects[0].subdims for op in self._objects), "Operators must share dimensions."
        assert all(not op.is_super for op in self._objects), "Cannot add superoperators."

    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None):
        return False

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return EvaluatedQuadruple(hamiltonian=super().partial_evaluate(t, parameters))


class EnvironmentBase(AQuantumSystem, TimeVectorOperator):
    """
    A vector of additional collapse operators or superoperators describing the environment, that may have
    time-dependent amplitudes.
    """

    def __init__(self,
                 operators: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        super().__init__(operators=operators, parameters=parameters, name=name, types=types)

    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None):
        return self.is_time_dependent(t, parameters)

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return EvaluatedQuadruple(environment=super().partial_evaluate(t, parameters))


class NaturalSystem(SystemCollection):
    """
    A quantum system whose evolution is determined by a HamiltonianBase and an EnvironmentBase, representing an uncontrolled
    or 'natural' system.
    """

    def __init__(self,
                 hamiltonian: Union[HamiltonianBase, OperatorInputList] = None,
                 environment: Union[EnvironmentBase, OperatorInputList] = None,
                 states: dict = None,
                 operators: dict = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):

        self.hamiltonian = hamiltonian if isinstance(hamiltonian, HamiltonianBase) else HamiltonianBase(hamiltonian)
        self.hamiltonian.default_name = '_hamiltonian'
        self.environment = environment if isinstance(environment, EnvironmentBase) else EnvironmentBase(environment)
        self.environment.default_name = '_environment'

        self.states = {} if states is None else states
        self._operators = {} if operators is None else operators

        super().__init__(parameters=parameters, name=name, rule=sum,
                         types=[HamiltonianBase, EnvironmentBase] if types is None else types)
        self._objects.append(self.hamiltonian)
        self._objects.append(self.environment)
        self._check_objects()

    def _check_objects(self):
        self._check_keys()
        if self.hamiltonian.subdims and self.environment.subdims:
            assert self.hamiltonian.subdims == self.environment.subdims, \
                "EnvironmentBase and HamiltonianBase must share the same dimensions"

    def _check_add(self, operator, parameters: dict = None, name: str = None):
        return super(TimeFunctionCollection, self)._check_add(operator, parameters, name)

    def _add(self, operator: Union[HamiltonianBase, EnvironmentBase], parameters: dict = None, name: str = None):
        if isinstance(operator, HamiltonianBase):
            self.hamiltonian.add(self._check_add(operator, parameters, name))
        elif isinstance(operator, EnvironmentBase):
            self.environment.add(self._check_add(operator, parameters, name))

    @property
    def subdims(self):
        return self.hamiltonian.subdims if self.hamiltonian.operator_list else self.environment.subdims

    @property
    def operators(self):
        return self._operators

    @operators.setter
    def operators(self, operators):
        self._operators = operators
