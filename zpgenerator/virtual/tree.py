from ..time.evaluate import EvaluatedDiracOperator
from .state import VState
from .propagator import AVirtualPropagator
from .branch import MeasurementBranch
from copy import deepcopy
from qutip import Qobj
from typing import Union
import numpy as np


class VNode:
    """
    An object of the VNode class is a node in a virtual tree containing a virtual state, and is defined at the beginning
     of at least one measurement time bin. It can be branched to give rise to nodes that contain states that propagate
     to future times.

    :param virtual_state: the virtual state of the node.
    :param future: a list of nodes corresponding to time in the future relative to this node.
    """

    def __init__(self, virtual_state: VState, future: list = None):
        self.virtual_state = virtual_state
        self.future = [] if future is None else future

    # Creates child nodes corresponding to a set of new configurations by copying the node virtual state
    def add_branch(self, branch: MeasurementBranch, pos: int = -1):
        if not self.future:
            for virtual_configuration in branch.virtual_configurations():
                virtual_state = deepcopy(self.virtual_state)
                branch_num = len(virtual_state.virtual_configuration)
                if pos >= branch_num:
                    virtual_state.virtual_configuration = virtual_state.virtual_configuration + [0] * (pos - branch_num + 1)
                elif branch_num == 0 and pos == -1:
                    virtual_state.virtual_configuration = [0]
                virtual_state.virtual_configuration[pos] = virtual_configuration
                self.future.append(VNode(virtual_state=virtual_state))
        else:
            for node in self.future:
                node.add_branch(branch, pos)

    def apply_operator(self, op: Union[Qobj, EvaluatedDiracOperator]):
        if not self.future:
            self.virtual_state.apply_operator(op)
        else:
            for node in self.future:
                node.apply_operator(op)

    def apply_generator(self, op: Qobj):
        if not self.future:
            self.virtual_state.apply_generator(op)
        else:
            for node in self.future:
                node.apply_generator(op)

    def propagate(self, propagator: AVirtualPropagator, t: float):
        if not self.future:
            self.virtual_state.propagate(propagator, t)
        else:
            for node in self.future:
                node.propagate(propagator, t)

    def get_states(self, states: list):
        if not self.future:
            states.append(self.virtual_state)
        else:
            for node in self.future:
                node.get_states(states)

    def get_points(self, states: list):
        if not self.future:
            states.append(self.virtual_state.tr())
        else:
            for node in self.future:
                node.get_points(states)

    def get_dimensions(self, dim: list, layer: int):
        if self.future:
            if len(dim) <= layer:
                dim += [0]
            dim[layer] = max(dim[layer], len(self.future))
            for node in self.future:
                node.get_dimensions(dim, layer + 1)

    def build_probability_tensor(self, tensor: np.array, coo: list):
        if not self.future:
            tensor[tuple(coo)] = self.virtual_state.tr()
        else:
            for j, node in enumerate(self.future):
                node.build_probability_tensor(tensor, coo + [j])

    def build_state_tensor(self, tensor: np.array, coo: list):
        if not self.future:
            state = self.virtual_state if self.virtual_state.isoper else self.virtual_state * self.virtual_state.dag()
            tensor[tuple(coo)] = np.asarray(state)
        else:
            for j, node in enumerate(self.future):
                node.build_state_tensor(tensor, coo + [j])


class VTree:
    """
    An object of the VTree class contains a tree structure of VNode objects representing the virtual state and
    virtual configuration chronolog.
    """

    def __init__(self, initial_state: VState):
        self.future = [VNode(virtual_state=initial_state)]
        self.branches = []
        self.subdims = initial_state.dims[0]

    @property
    def branch_number(self):
        return len(self.branches)

    def add_branch(self, branch: MeasurementBranch, pos: int = -1):
        for node in self.future:
            node.add_branch(branch, pos)
            self.branches.append(branch)

    def apply_operator(self, op: Union[Qobj, EvaluatedDiracOperator]):
        for node in self.future:
            node.apply_operator(op)

    def apply_generator(self, op: Qobj):
        for node in self.future:
            node.apply_generator(op)

    def propagate(self, propagator: AVirtualPropagator, t: float):
        for node in self.future:
            node.propagate(propagator, t)

    def get_states(self):
        states = []
        for node in self.future:
            node.get_states(states)
        return states

    def get_points(self):
        points = []
        for node in self.future:
            node.get_points(points)
        return points

    def get_dimensions(self):
        dim = []
        for node in self.future:
            node.get_dimensions(dim, 0)
        return dim

    def build_probability_tensor(self):
        tensor = np.empty(self.get_dimensions(), dtype=complex)
        coo = []
        for node in self.future:
            node.build_probability_tensor(tensor, coo)
        return tensor

    def build_state_tensor(self):
        tensor = np.empty(self.get_dimensions(), dtype=object)
        coo = []
        for node in self.future:
            node.build_state_tensor(tensor, coo)
        return self._convert_to_numeric(tensor)

    def _convert_to_numeric(self, tensor):
        if isinstance(tensor, np.ndarray) and tensor.dtype == object:
            return np.stack([self._convert_to_numeric(elem) for elem in tensor])
        else:
            return tensor
