from ..time.evaluate import EvaluatedDiracOperator
from .inverse import GeneratingTensor
from .branch import MeasurementBranch
from .tree import VTree
from .state import VState
from .propagator import AVirtualPropagator
from typing import List, Union
from qutip import Qobj


class VGrove:

    def __init__(self, initial_time: float, states: List[Qobj]):
        self.trees = [VTree(initial_state=VState(state=state, time=initial_time)) for state in states]
        self.time = initial_time

    def __iter__(self):
        return iter(self.trees)

    def _add_branches(self, time: float, branches: List[MeasurementBranch], initial_branches: bool = False):
        branch_order = []
        for k, tree in enumerate(self):
            for i, branch in enumerate(branches):
                if branch.start_time < time if initial_branches else branch.start_time == time:
                    tree.add_branch(branch, i)
                    if k == 0:
                        branch_order.append(i)
        return branch_order

    def initialize(self, time: float, branches: List[MeasurementBranch]):
        return self._add_branches(time, branches, initial_branches=True)

    def add_branches(self, time: float, branches: List[MeasurementBranch]):
        return self._add_branches(time, branches)

    def apply_operator(self, op: Union[Qobj, EvaluatedDiracOperator]):
        for tree in self:
            tree.apply_operator(op)

    def apply_generator(self, op: Qobj):
        for tree in self:
            tree.apply_generator(op)

    def propagate(self, propagator: AVirtualPropagator, time: float):
        for tree in self:
            tree.propagate(propagator, time)

    def build_tensors(self, point_rank: int, precision: int):
        return [GeneratingTensor(point_rank, tree, precision) for tree in self]
