from ..time import ATimeOperator, TimeFunctionCollection, TimeOperatorCollection, EvaluatedOperator, id_flatten, \
    DefaultCache
from ..time.evaluate.quadruple import EvaluatedQuadruple
from abc import abstractmethod
from typing import Union, List
from qutip import Qobj


class AQuantumSystem(ATimeOperator):
    """
    A generic nonlinear system object that can be used to build a master equation generator.
    """

    @property
    def initial_state(self):
        return None

    @property
    def initial_time(self):
        return None

    @property
    def modes(self) -> int:
        return 0

    @abstractmethod
    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None):
        pass

    @abstractmethod
    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        pass

    def evaluate(self, t: float, parameters: dict = None) -> Qobj:
        """
        :return: a Liouvillian (generator)
        """
        return self.evaluate_quadruple(t, parameters).evaluate(t)


class SystemCollection(AQuantumSystem, TimeOperatorCollection):
    """
    A collection of quantum systems with a rule to evaluate
    """

    def __init__(self,
                 systems: Union[AQuantumSystem, List[AQuantumSystem]] = None,
                 parameters: dict = None,
                 name: str = None,
                 rule: callable = sum,
                 types: list = None):
        super().__init__(operators=systems, parameters=parameters, name=name, rule=rule,
                         types=[AQuantumSystem] if types is None else types)

    def _check_add(self, operator, parameters: dict = None, name: str = None):
        return super(TimeFunctionCollection, self)._check_add(operator, parameters, name)

    @DefaultCache(time_arg=True)
    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        return any(system.is_nonhermitian_time_dependent(t, parameters) for system in self._objects)

    @DefaultCache(time_arg=True)
    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        parameters = self.set_parameters(parameters)
        return self._rule([system.evaluate_quadruple(t, parameters) for system in self._objects], EvaluatedQuadruple())

    def partial_evaluate(self, t: float, parameters: dict = None) -> List[EvaluatedOperator]:
        parameters = self.set_parameters(parameters)
        return id_flatten([system.partial_evaluate(t, parameters) for system in self._objects])

