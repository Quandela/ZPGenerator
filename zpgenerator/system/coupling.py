from ..time import TimeOperator, TimeOperatorCollection, evop_tensor_flatten, OperatorInputList
from .natural import HamiltonianBase
from qutip import Qobj
from typing import Union, List
from math import prod


class CouplingTerm(TimeOperatorCollection):
    """
    An object that defines a collection of operators to be tensored together
    """

    def __init__(self,
                 operators: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None):
        super().__init__(operators=operators, parameters=parameters, name=name, rule=evop_tensor_flatten)

    def _check_objects(self):
        self._check_keys()  # self.operators is mutable, children updated automatically, only check_keys left to do
        assert all(not op.is_super for op in self._objects), "Cannot add superoperators."

    @property
    def bodies(self):
        return len(self._objects)

    @property
    def dim(self) -> int:
        return prod(op.dim for op in self._objects) if self._objects else None

    @property
    def subdims(self):
        dimset = [op.subdims for op in self._objects]
        return [dim for dims in dimset for dim in dims]

    def insert_dimension(self, pos: int, dim: Union[int, List[int]]):
        self._objects.insert(pos, TimeOperator.identity(dim))

    def pad_left(self, dim: Union[int, List[int]]):
        dims = dim if isinstance(dim, list) else [dim]
        self._objects = [TimeOperator.identity(dim) for dim in dims] + self._objects

    def pad_right(self, dim: Union[int, List[int]]):
        dims = dim if isinstance(dim, list) else [dim]
        self._objects = self._objects + [TimeOperator.identity(dim) for dim in dims]


class CouplingBase(HamiltonianBase):
    """
    An object that defines a collection of coupling terms that are summed together
    """

    def __init__(self,
                 operators: Union[CouplingTerm, List[CouplingTerm]] = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        super().__init__(operators=operators, parameters=parameters, name=name,
                         types=[HamiltonianBase, CouplingTerm, CouplingBase] if types is None else types)

    def _check_objects(self):
        super()._check_objects()
        assert all(op.bodies == self.bodies for op in self._objects), "Number of bodies must be the same for all terms"

    @property
    def bodies(self):
        return self._objects[0].bodies if self._objects else 0

    def insert_dimension(self, pos: int, dim: Union[int, List[int]]):
        for term in self._objects:
            term.insert_dimension(pos, dim)

    def pad_left(self, dim: Union[int, List[int]]):
        for term in self._objects:
            term.pad_left(dim)

    def pad_right(self, dim: Union[int, List[int]]):
        for term in self._objects:
            term.pad_right(dim)

    @classmethod
    def jaynes_cummings(cls, operator0: Qobj, operator1: Qobj):
        term0 = CouplingTerm([lambda args: args['coupling'] * operator0, operator1.dag()], parameters={'coupling': 1})
        term1 = CouplingTerm([lambda args: args['coupling'] * operator0.dag(), operator1], parameters={'coupling': 1})
        return cls([term0, term1])
