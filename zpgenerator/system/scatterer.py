from ..time import ATimeOperator, EvaluatedOperator, TimeOperator, CompositeTimeOperator, \
    TimeOperatorCollection, OperatorInputList
from ..time.evaluate.quadruple import EvaluatedQuadruple
from abc import abstractmethod
from qutip import rand_unitary_haar
from math import prod


class AElement(ATimeOperator):
    """
    A parameterized object that can be used to make a component
    """

    @property
    @abstractmethod
    def modes(self) -> int:
        pass

    @property
    def input_modes(self) -> int:
        return self.modes

    @property
    def output_modes(self) -> int:
        return self.modes

    @property
    @abstractmethod
    def is_emitter(self) -> bool:
        pass

    @abstractmethod
    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        pass


class AScatteringMatrix(AElement):

    @property
    def is_emitter(self) -> bool:
        return False

    @classmethod
    @abstractmethod
    def haar_random(cls, n: int):
        pass


class ScattererBase(AScatteringMatrix, CompositeTimeOperator):
    "An arbitrary scattering matrix constructed by summing matrices that may have time dependent scalar functions"

    def __init__(self,
                 matrices: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None):
        """
        :param matrices: a list of scattering matrices.
        :param parameters: a list of parameters that will set the default parameters for the system.
        :param name: the optional name of the system to distinguish it from other similar nonlinear.
        """
        self.matrices = matrices
        super().__init__(operators=matrices, parameters=parameters, name=name)

    @property
    def modes(self):
        return self.dim

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return EvaluatedQuadruple(
            transitions=[EvaluatedOperator()] * self.modes,
            scatterer=self.partial_evaluate(t, parameters))

    @classmethod
    def haar_random(cls, n: int):
        return cls(TimeOperator(rand_unitary_haar(n)))


class MultiScatterer(AScatteringMatrix, TimeOperatorCollection):
    "An arbitrary scattering matrix constructed by matrix multiplication"

    def __init__(self,
                 matrices: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None):
        """
        :param matrices: the scattering matrix.
        :param parameters: a list of parameters that will set the default parameters for the system.
        :param name: the optional name of the system to distinguish it from other similar nonlinear.
        """
        self.matrices = matrices
        super().__init__(operators=matrices, parameters=parameters, name=name,
                         rule=lambda objects, *args: prod(objects))

    @property
    def modes(self):
        return self.dim

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return EvaluatedQuadruple(
            transitions=[EvaluatedOperator()] * self.modes,
            scatterer=self.partial_evaluate(t, parameters))

    @classmethod
    def haar_random(cls, n: int):
        return cls(TimeOperator(rand_unitary_haar(n)))


class CompositeScatterer(AScatteringMatrix, TimeOperatorCollection):
    "An arbitrary scattering matrix constructed by matrix summation"

    def __init__(self,
                 matrices: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None):
        """
        :param matrices: the scattering matrix.
        :param parameters: a list of parameters that will set the default parameters for the system.
        :param name: the optional name of the system to distinguish it from other similar nonlinear.
        """
        self.matrices = matrices
        super().__init__(operators=matrices, parameters=parameters, name=name)

    @property
    def modes(self):
        return self.dim

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return EvaluatedQuadruple(
            transitions=[EvaluatedOperator()] * self.modes,
            scatterer=self.partial_evaluate(t, parameters))

    @classmethod
    def haar_random(cls, n: int):
        return cls(TimeOperator(rand_unitary_haar(n)))
