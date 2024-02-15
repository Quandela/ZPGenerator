from .evaluate.tensor import id_flatten
from .function import ATimeFunction, TimeFunction, TimeInstantFunction, TimeInstant
from .operator import AOperator, Operator
from .composite import TimeFunctionCollection, CompositeTimeFunction
from .evaluate import OpFuncPair, EvaluatedOperator, Func
from .evaluate.dirac import EvaluatedDiracOperator
from abc import abstractmethod
from typing import Union, List, Callable
from qutip import Qobj
from numpy import ndarray
from inspect import signature

OperatorInputs = Union[AOperator, Qobj, List[list], Callable, ndarray]
OperatorInputList = Union[OperatorInputs, List[OperatorInputs]]


class ATimeOperator(ATimeFunction, AOperator):
    """
    A function of time on a domain that returns a Qobj operator or superoperator.
    """

    @abstractmethod
    def partial_evaluate(self, t: float, parameters: dict = None) -> Union[EvaluatedOperator, List[EvaluatedOperator]]:
        """
        :param t: a real value.
        :param parameters: a dictionary of arguments used to evaluate the operator.
        :return: an EvaluatedOperator object
        """
        pass

    @abstractmethod
    def evaluate(self, t: float, parameters: dict = None) -> Union[Qobj, List[Qobj]]:
        """
        :param t: a real value.
        :param parameters: a dictionary of arguments used to evaluate the operator.
        :return: a Qobj
        """
        pass

    @abstractmethod
    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        """
        :param t: a real value.
        :param parameters: a dictionary of arguments used to evaluate the operator.
        :return: a Qobj
        """
        pass


class TimeOperator(CompositeTimeFunction, ATimeOperator):
    """
    A single operator or superoperator modulated by a composite time function.
    """

    def __init__(self,
                 operator: Union[OperatorInputs, ndarray, list],
                 functions: Union[ATimeFunction, List[ATimeFunction]] = None,
                 parameters: dict = None,
                 name: str = None,
                 dag: bool = False):
        """
        :param operator: a qutip Qobj or a function f(parameters) that returns a Qobj.
        :param functions: a time function or list of functions defining the modulation function.
        :param parameters: a dictionary of parameters needed to evaluate the operator and modulation functions.
        :param name: a name for the object to distinguish parameters.
        """
        self.operator = operator if isinstance(operator, AOperator) else \
            Operator(matrix=operator, parameters=parameters if callable(operator) else None)

        self._init_empty = True if functions is None else False
        super().__init__(functions=TimeFunction(value=1) if self._init_empty else functions,
                         name=name, parameters=parameters)
        self.set_children([self._objects, self.operator])
        self._dag = dag
        self._check_objects()

    def _check_add(self, operator, parameters: dict = None, name: str = None):
        if self._init_empty and self._objects:
            self._objects.pop()
            self._init_empty = False
        return super()._check_add(operator, parameters, name)

    @property
    def is_super(self) -> bool:
        return self.operator.is_super

    @property
    def dim(self) -> int:
        return self.operator.dim

    @property
    def subdims(self) -> List[int]:
        return self.operator.subdims

    def partial_evaluate(self, t: float, parameters: dict = None) -> EvaluatedOperator:
        op = self.operator.evaluate(self.set_parameters(parameters))
        if self.is_time_dependent(t, parameters):
            parameters = self.get_parameters(parameters)
            op = EvaluatedOperator(variable=OpFuncPair(op=op, func=self.evaluate_function(t, parameters),
                                                       parameters=parameters))
        else:
            op = EvaluatedOperator(constant=super().evaluate(t, parameters) * op)
        return op.dag() if self._dag else op

    def evaluate(self, t: float, parameters: dict = None) -> EvaluatedOperator:
        op = self.partial_evaluate(t, parameters).evaluate(t)
        return op.dag() if self._dag else op

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        op = super().evaluate_dirac(t, parameters) * self.operator.evaluate(self.set_parameters(parameters))
        op = EvaluatedDiracOperator(channel=op) if op.issuper else EvaluatedDiracOperator(hamiltonian=op)
        return op.dag() if self._dag else op

    @classmethod
    def identity(cls, dim: int):
        return cls(operator=Operator.identity(dim))

    @classmethod
    def dirac(cls, operator, time: Union[float, str], parameters: dict = None, name: str = None):
        return TimeOperator(operator=operator,
                            functions=TimeInstantFunction(value=1, instant=TimeInstant([time], parameters)),
                            parameters=parameters,
                            name=name)


# Grabbing common properties of CompositeTimeOperator and TimeVectorOperator together to reduce duplication
class TimeOperatorCollection(TimeFunctionCollection, ATimeOperator):
    """A collection of time operators with the same structure that are evaluated following a defined rule"""

    def __init__(self,
                 operators: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None,
                 rule: callable = sum,
                 types: list = None):
        """
        :param operators: a list of time function operators.
        :param parameters: a dictionary of parameters needed to evaluate the time operators.
        :param name: a name for the object to distinguish parameters.
        :param rule: a function applied to the list of operators when evaluating the collection
        """
        self._initial_parameters = parameters
        super().__init__(functions=operators, parameters=parameters, name=name, rule=rule,
                         types=[AOperator, Qobj, Callable, float, int, complex] if types is None else types)

    @property
    def operator_list(self) -> list:
        return self._objects

    @property
    def operators(self) -> dict:
        keys = self._make_unique_names()
        return {keys[i]: op for i, op in enumerate(self._objects)}

    def _check_objects(self):
        self._check_keys()
        assert all(op.subdims == self._objects[0].subdims for op in self._objects), \
            "Operators must share dimensions."

    def _check_add(self, operator, parameters: dict = None, name: str = None):
        operator = super(TimeFunctionCollection, self)._check_add(operator, parameters, name)
        if isinstance(operator, AOperator):
            if isinstance(operator, ATimeOperator):
                op = operator
            else:
                op = TimeOperator(operator=operator)
        elif callable(operator):
            if len(signature(operator).parameters) == 1:
                op = TimeOperator(operator=Operator(matrix=operator,
                                                    parameters=self._initial_parameters if self._initial_parameters else parameters,
                                                    name=name))
                self._initial_parameters = None
            else:
                op = TimeOperator(operator=Qobj(1))
                op.add(TimeFunction(value=operator, parameters=parameters, name=name))
        else:
            op = TimeOperator(operator=operator, parameters=parameters, name=name)

        if hasattr(op, 'operator'):
            op.operator.default_name = '_operator'
        return op

    @property
    def dim(self) -> int:
        return self._objects[0].dim if self._objects else None

    @property
    def subdims(self) -> list:
        return self._objects[0].subdims if self._objects else None

    @property
    def is_super(self) -> bool:
        return any(op.is_super for op in self._objects)

    @property
    def has_instant(self) -> bool:
        return any(op.has_instant for op in self._objects)

    @property
    def has_interval(self) -> bool:
        return any(op.has_interval for op in self._objects)

    def partial_evaluate(self, t: float, parameters: dict = None) -> Union[EvaluatedOperator, List[EvaluatedOperator]]:
        parameters = self.set_parameters(parameters)
        return self._rule([op.partial_evaluate(t, parameters) for op in self._objects], EvaluatedOperator())

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        parameters = self.set_parameters(parameters)
        return self._rule(id_flatten([op.evaluate_dirac(t, parameters) for op in self._objects]),
                          EvaluatedDiracOperator())


class CompositeTimeOperator(TimeOperatorCollection):
    """
    A set of time operators that sum together allowing for multiple time-dependent elements.
    """

    def __init__(self,
                 operators: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        """
        :param operators: a list of time function operators.
        :param parameters: a dictionary of parameters needed to evaluate the time operators.
        :param name: a name for the object to distinguish parameters.
        """
        super().__init__(operators=operators, parameters=parameters, name=name, rule=sum, types=types)

    def _check_objects(self):
        self._check_keys()  # self._objects is mutable, children updated automatically, only check_keys left to do
        assert all(op.subdims == self._objects[0].subdims for op in self._objects), "Operators must share dimensions."
        assert all(op.is_super == self._objects[0].is_super for op in self._objects), "Operators must share is_super."

    @classmethod
    def identity(cls, dim: int):
        return cls(operators=Operator.identity(dim))


class TimeVectorOperator(TimeOperatorCollection):

    def __init__(self,
                 operators: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        """
        :param operators: a list of time function operators.
        :param parameters: a dictionary of parameters needed to evaluate the time operators.
        :param name: a name for the object to distinguish parameters.
        """
        super().__init__(rule=id_flatten, operators=operators, parameters=parameters, name=name, types=types)

    @property
    def length(self) -> int:
        return len(self._objects)
