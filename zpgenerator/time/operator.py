# Classes defining parameterised operators and time-dependent operators

from .parameters.parameterized_object import AParameterizedObject, ParameterizedObject
from .parameters.collection import ParameterizedCollection
from abc import abstractmethod, ABC
from typing import Union, List, Callable
from qutip import Qobj, qeye
from numpy import sin, cos, exp, ndarray
from copy import deepcopy


class AOperator(AParameterizedObject):
    """An object that is evaluated to an operator, perhaps with some optional arguments."""

    @abstractmethod
    def evaluate(self, parameters: dict = None) -> Qobj:
        """
        Evaluates the operator.

        :param parameters: a dictionary of arguments used to evaluate the domain.
        :return: a time instant or interval, or list of instants or intervals.
        """
        pass

    @property
    @abstractmethod
    def is_super(self) -> bool:
        """
        Whether the operator is a superoperator.

        :return: a boolean
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        The Hilbert space dimension of the operator (matrix size).

        :return: a positive integer
        """
        pass

    @property
    @abstractmethod
    def subdims(self) -> List[int]:
        """
        A list of Hilbert space dimensions specifying the underlying tensor product structure of the operator.

        :return: a list of positive integers
        """
        pass


class Operator(ParameterizedObject, AOperator):
    """A quantity that evaluates to a Qobj when provided with some arguments"""

    def __init__(self,
                 matrix: Union[callable, Qobj, List[list], ndarray],
                 parameters: dict = None,
                 name: str = None):
        """
        :param matrix: a function f(parameters) -> Qobj or a Qobj.
        :param parameters: a dictionary of parameters needed to evaluate the operator function.
        :param name: a name for the object to distinguish parameters.
        """
        super().__init__(parameters=parameters, name=name)
        self.matrix = Qobj(inpt=matrix) if isinstance(matrix, list) or isinstance(matrix, ndarray) else matrix
        self.is_callback = callable(matrix) and not isinstance(matrix, Qobj)
        self.scale = 1
        self._check_operator()

    def _check_operator(self):
        test = self.evaluate()
        assert isinstance(test, Qobj), "Must evaluate to a Qobj"
        shape = test.shape
        assert shape[0] == shape[1], "Must be square."
        self._is_super = test.issuper
        self._dim = shape[0]
        self._subdims = test.dims[0][0] if self._is_super else test.dims[0]

    def evaluate(self, parameters: dict = None) -> Qobj:
        parameters = self.get_parameters(parameters)
        matrix = self.matrix(parameters) if self.is_callback else self.matrix
        if not isinstance(matrix, Qobj):
            matrix = Qobj(matrix)
        return matrix * self.scale

    @property
    def is_super(self) -> bool:
        return self._is_super

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def subdims(self) -> List[int]:
        return self._subdims

    @classmethod
    def polarised(cls, operator1: Union[callable, Qobj], operator2: Union[callable, Qobj],
                  parameters: dict = None, name: str = None,
                  angle_str: str = 'theta', phase_str: str = 'phi'):
        parameters = {} if parameters is None else parameters

        def op(args: dict):
            op1 = operator1 if isinstance(operator1, Qobj) else operator1(args)
            op2 = operator2 if isinstance(operator2, Qobj) else operator2(args)
            return cos(args[angle_str]) * op1 + exp(1.j * args[phase_str]) * sin(args[angle_str]) * op2

        return cls(matrix=op, parameters={angle_str: 0, phase_str: 0} | parameters, name=name)

    @classmethod
    def polarised_orthogonal(cls, operator1: Union[callable, Qobj], operator2: Union[callable, Qobj],
                  parameters: dict = None, name: str = None,
                  angle_str: str = 'theta', phase_str: str = 'phi'):
        parameters = {} if parameters is None else parameters

        def op(args: dict):
            op1 = operator1 if isinstance(operator1, Qobj) else operator1(args)
            op2 = operator2 if isinstance(operator2, Qobj) else operator2(args)
            return sin(args[angle_str]) * op1 - exp(1.j * args[phase_str]) * cos(args[angle_str]) * op2

        return cls(matrix=op, parameters={angle_str: 0, phase_str: 0} | parameters, name=name)

    @classmethod
    def identity(cls, dim: Union[int, List[int]]):
        return cls(matrix=qeye(dim))


class AOperatorCollection(ParameterizedCollection, AOperator, ABC):
    """
    A collection of operators
    """

    def __init__(self,
                 operators,
                 parameters: dict = None,
                 name: str = None,
                 rule: callable = sum):
        self._rule = rule
        super().__init__(objects=operators, parameters=parameters, name=name,
                         types=[AOperator, Callable, List[list], ndarray, Qobj])

    def _check_add(self, operator, parameters: dict = None, name: str = None):
        if isinstance(operator, AOperator):
            if parameters or name:
                op = deepcopy(operator)
                op.update_default_parameters(parameters)
                op.name = name
            else:
                op = operator
            return op
        else:
            name = '_operator ' + ''.join(['(', str(len(self._objects)), ')'])
            return Operator(matrix=operator, parameters=parameters, name=name)

    @property
    def operators(self) -> dict:
        keys = self._make_unique_names()
        return {keys[i]: comp for i, comp in enumerate(self._objects)}

    def evaluate(self, parameters: dict = None) -> Qobj:
        parameters = self.set_parameters(parameters)
        return self._rule(operator.evaluate(parameters) for operator in self._objects)


class CompositeOperator(AOperatorCollection):
    """
    A collection of operators with summation
    """
    def __init__(self, operators=None, parameters: dict = None, name: str = None):
        super().__init__(operators=operators, parameters=parameters, name=name, rule=sum)

    def _check_objects(self):
        self._check_keys()
        if self._objects:
            assert all(operator.subdims == self._objects[0].subdims for operator in self._objects), \
                "All operators must share dimensions"
        if self._objects:
            assert all(operator.is_super == self._objects[0].is_super for operator in self._objects), \
                "All operators must share is_super."

    @property
    def is_super(self) -> bool:
        return self._objects[0].is_super if self._objects else None

    @property
    def dim(self) -> int:
        return self._objects[0].dim if self._objects else None

    @property
    def subdims(self) -> List[int]:
        return self._objects[0].subdims if self._objects else None
