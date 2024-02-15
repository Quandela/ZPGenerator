from ..time import TimeFunctionCollection, TimeOperatorCollection, EvaluatedDiracOperator, EvaluatedQuadruple, \
    id_flatten, TupleDict
from ..system import AElement, AScatteringMatrix, AQuantumSystem, AQuantumEmitter
from typing import List, Union
from math import prod
from qutip import tensor, Qobj


class ElementCollection(AElement, TimeOperatorCollection):
    """
    A collection of elements in a quantum component evaluated in a quantum cascaded
    """

    def __init__(self,
                 elements: Union[AElement, List[AElement]] = None,
                 parameters: dict = None,
                 name: str = None,
                 rule: callable = None,
                 types: list = None):
        self._elements = []
        super().__init__(operators=elements, parameters=parameters, name=name,
                         rule=lambda objects, *args: prod(objects) if rule is None else rule,
                         types=[AElement] if types is None else types)

        self._initial_state = None
        self._initial_time = None

    def _check_objects(self):
        super(TimeFunctionCollection, self)._check_objects()
        if self._objects:
            assert all(element.modes == self._objects[0].modes for element in self._objects), \
                "All elements must share the same number of modes"

    def _check_add(self, element, parameters: dict = None, name: str = None):
        element = super()._check_add(element, parameters, name)
        self._elements.append(element)
        return element

    @property
    def state_list(self) -> list:
        return [element.states for element in self._elements if element.is_emitter]

    @property
    def state_labels(self) -> list:
        return [[k for k in states.keys()] for states in self.state_list]

    @property
    def states(self) -> TupleDict:
        state_list = self.state_list
        states = TupleDict(state_list[0])
        for substates in state_list[1:]:
            states = self._combine_state_dictionaries(states, substates)
        return states

    def _combine_state_dictionaries(self, states0: TupleDict, states1: TupleDict) -> TupleDict:
        new_states = TupleDict()
        for k0, v0 in states0.items():
            for k1, v1 in states1.items():
                new_states.update({k0 + k1: tensor(v0, v1)})
        return new_states

    @property
    def initial_state(self):
        if self._initial_state:
            return self._initial_state
        else:
            initial_states = [system.initial_state for system in self._objects if hasattr(system, 'initial_state')
                              and system.initial_state != 1]
            if any(initial_state.isoper for initial_state in initial_states):
                initial_states = [state * state.dag() if not state.isoper else state for state in initial_states]
            if any(initial_state is None for initial_state in initial_states):
                return None
            elif not initial_states:
                return 1
            else:
                return tensor(initial_states) if len(initial_states) > 1 else initial_states[0]

    @initial_state.setter
    def initial_state(self, state: Qobj):
        self._initial_state = state

    @property
    def initial_time(self):
        if self._initial_time is not None:
            return self._initial_time
        else:
            initial_times = [system.initial_time for system in self._objects if hasattr(system, 'initial_time') and
                             system.initial_time is not None]
            if not all(times == initial_times[0] for times in initial_times):
                print("Warning: initial times for one or more elements disagree, "
                      "proceeding by taking the earliest time.")
            return min(initial_times) if initial_times else None

    @initial_time.setter
    def initial_time(self, time: Union[float, int]):
        self._initial_time = time

    @property
    def input_modes(self) -> int:
        return self.modes

    @property
    def output_modes(self) -> int:
        return self.modes

    @property
    def dim(self) -> int:
        return prod([element.dim for element in self._elements if isinstance(element, AQuantumSystem)])

    @property
    def subdims(self) -> List[int]:
        subdims = []
        for element in self._elements:
            if element.is_emitter:
                subdims += element.subdims
        return subdims

    @property
    def elements(self) -> dict:
        keys = self._make_unique_names()
        return {keys[i]: comp for i, comp in enumerate(self._elements)}

    @property
    def modes(self):
        return self._elements[0].modes if self._elements else 0

    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None):
        return any(element.is_nonhermitian_time_dependent(t, parameters)
                   if not isinstance(element, AScatteringMatrix) else
                   element.is_time_dependent(t, parameters) for element in self._elements)

    def evaluate_quadruple(self, t: float, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        if self._elements:
            return self._rule([element.evaluate_quadruple(t, parameters) for element in self._elements])
        else:
            return EvaluatedQuadruple()

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        parameters = self.set_parameters(parameters)
        return self._rule(id_flatten([op.evaluate_dirac(t, parameters) for op in self._objects
                                      if isinstance(op, AQuantumEmitter) or isinstance(op, ElementCollection)]),
                          EvaluatedDiracOperator())


    @property
    def is_emitter(self):
        return any(element.is_emitter for element in self._elements)
