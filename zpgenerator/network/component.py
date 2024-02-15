from ..time import TimeFunctionCollection, merge_times, EvaluatedQuadruple
from ..time.evaluate.cache import DefaultCache
from ..system import AElement, ScattererBase, AScatteringMatrix
from .element import ElementCollection
from .detector import ADetectorGate
from .port import InputLayer, OutputLayer, OutputPort, InputPort
from typing import Union, List
from abc import abstractmethod
from copy import deepcopy
from qutip import Qobj
from frozendict import frozendict


class AComponent(ElementCollection):
    """A collection of one or more elements with a layer of input and output ports"""

    @property
    @abstractmethod
    def input(self) -> InputLayer:
        pass

    @property
    @abstractmethod
    def output(self) -> OutputLayer:
        pass

    def bin_all_detectors(self, bin_name: str):
        self.output.bin_all_detectors(bin_name)

    @property
    def input_modes(self) -> int:
        return self.input.open_modes

    @property
    def output_modes(self) -> int:
        return self.output.open_modes

    @property
    def type(self) -> str:
        if self.input.is_closed and self.output.is_closed:
            return 'Processor'
        elif not self.output.is_open and self.output.is_monitored:
            return 'Detector'
        elif self.input.is_closed:
            return 'Source'
        elif self.output.is_closed:
            return 'Trace'
        else:
            return 'Element'

    @abstractmethod
    def mask(self):
        pass

    @abstractmethod
    def unmask(self):
        pass

    @property
    @abstractmethod
    def is_masked(self) -> bool:
        pass


class Component(AComponent):
    """
    A collection of elements with some inputs and outputs having different statuses
    """

    ComponentInputTypes = Union[AElement, List[AElement], ADetectorGate, List[ADetectorGate]]

    def __init__(self,
                 elements: ComponentInputTypes = None,
                 parameters: dict = None,
                 name: str = None,
                 masked: bool = True,
                 types: list = None):
        self._is_masked = masked
        self._input = InputLayer.make(0)
        self._output = OutputLayer.make(0)
        self.permutations = []
        self._next_pos = 0

        super().__init__(parameters=parameters, name=name, types=[AElement, ADetectorGate] if not types else types)
        self.set_children([self._objects, self._output.ports])
        if isinstance(elements, list):
            for elm in elements:
                self.add(elm)
        else:
            self.add(elements)

    def _check_objects(self):
        super(TimeFunctionCollection, self)._check_objects()

    @property
    def input(self) -> InputLayer:
        return self._input

    @property
    def output(self) -> OutputLayer:
        return self._output

    @property
    def modes(self):
        return len(self.permutations[0]) if self.permutations else 0

    @property
    def is_masked(self) -> bool:
        return self._is_masked

    @is_masked.setter
    def is_masked(self, value):
        self._is_masked = value

    def mask(self):
        self._is_masked = True

    def unmask(self):
        self._is_masked = False

    def unmasked_position(self, position: int):
        """
        :param position: a masked (not closed) or unmasked position
        :return: the corresponding unmasked position
        """
        if self._elements and self.is_masked:
            try:
                return [i for i, port in enumerate(self.output.ports) if not port.is_closed][position]
            except IndexError:
                return self.modes
        else:
            return position

    def masked_position(self, position: int):
        """
        :param position: a masked (not closed) or unmasked position
        :return: the corresponding unmasked position
        """
        if self._elements and self.is_masked:
            return position
        else:
            return [port.is_closed for i, port in enumerate(self.output.ports)
                    if i < self.unmasked_position(position)].count(False)

    def get_port_number(self, position: Union[str, int]) -> int:
        if isinstance(position, str):
            if self.elements:
                number = self.output.get_port_number(position)
                assert number, "No port named " + position + "."
                position = number
            else:
                assert False, "Processor has no ports."
        umasked_position = self.unmasked_position(position)
        if self.is_masked and self._elements and umasked_position < self.modes:
            assert not self.output.ports[umasked_position].is_closed, \
                "Selected port to connect must not be closed."
            return self.masked_position(position)
        else:
            return position

    def _position_to_add(self, position: int):
        """
        :param position: a masked or unmasked position
        :return: the corresponding masked position, excluding monitored ports
        """
        if not self._elements or position >= self.modes:
            return position
        else:
            return [port.is_open for i, port in enumerate(self.output.ports)
                    if i < self.unmasked_position(position)].count(True)

    # annoying workaround to make signature of networks similar to Perceval while keeping other collections consistent
    # it is silly that Perceval doesn't consider 'position' a keyword argument in the second argument position
    def add(self,
            position: Union[int, str, ComponentInputTypes],
            element: ComponentInputTypes = None,
            parameters: dict = None, name: str = None, bin_name: str = None):

        if isinstance(position, int) or isinstance(position, str):
            assert element, "Please specify an element to add"
        else:
            element = position
            position = 0

        if hasattr(element, 'compute_unitary'):
            element = Component(ScattererBase(Qobj(element.compute_unitary())))

        position = self.get_port_number(position)

        if isinstance(element, AElement):
            if isinstance(element, AComponent) and bin_name:
                element = deepcopy(element)
                element.bin_all_detectors(bin_name)
                if parameters or name:  # do this here so we don't deepcopy twice
                    element.update_default_parameters(parameters)
                    parameters = None
                    element.name = name
                    name = None
            self._add_element(element, position, parameters, name)
        elif isinstance(element, ADetectorGate):
            self._add_detector(element, position, parameters, name, bin_name)

    def _add_element(self, element: AElement, position: int = None, parameters: dict = None, name: str = None):
        if self._elements:
            unmasked_position = self.unmasked_position(position)
            if unmasked_position < self.modes:
                assert self.output.ports[unmasked_position].is_open, \
                    "Selected port to connect must be open."
        position = self._position_to_add(position)  # maps the input position (masked or unmasked) to an open position
        self._next_pos = position
        super().add(element, parameters, name)

    def _add_detector(self, element: ADetectorGate, position: int = None,
                      parameters: dict = None, name: str = None, bin_name: str = None):
        position = self.unmasked_position(position)
        if self.elements:
            assert not self.output.ports[position].is_closed, "Selected port to connect must not be closed."
        self._output.ports[position].add(element, parameters=parameters, name=name, bin_name=bin_name)

    def __floordiv__(self, other):
        if isinstance(other, tuple):
            if len(other) == 2:
                position = other[0]
                element = other[1]
            else:
                position = 0
                element = other[0]
        else:
            position = 0
            element = other
        self.add(position, element)
        return self

    def _check_add(self, element, parameters: dict = None, name: str = None):
        if isinstance(element, AElement):
            self._connect(element)
        return super()._check_add(element, parameters, name)

    def _connect(self, element):
        """
        Connects open input ports of an element to the open outputs ports of a component
        :param element: an element to connect
        """
        connected_modes = min(self.output_modes, element.input_modes) - self._next_pos
        new_mode_number = self.modes + element.modes - connected_modes
        position_counter = self._next_pos

        if isinstance(element, AComponent):
            element_active_port = []
            element_inactive_port = []
            for i, port in enumerate(element.input.ports):
                port_to_add = element.output.ports[i]
                if port.is_closed:
                    element_inactive_port.append([i, port, port_to_add])
                else:
                    element_active_port.append([i, port, port_to_add])
            element_active_port = iter(element_active_port)
            element_inactive_port = iter(element_inactive_port)
        else:
            element_active_port = iter([[i, InputPort(), OutputPort()] for i in range(element.modes)])
            element_inactive_port = iter([])

        new_output = OutputLayer()
        new_input = InputLayer()

        perm = []
        element_extra_mode_number = iter(range(element.modes, new_mode_number))

        for i in range(self.modes):
            new_input.add(deepcopy(self.input.ports[i]))
            output_port = deepcopy(self.output.ports[i])
            if output_port.is_open:
                if position_counter:
                    new_output.add(output_port)  # treat as closed
                    perm.append(next(element_extra_mode_number))
                    position_counter -= 1
                else:
                    try:
                        port = next(element_active_port)
                        new_output.add(deepcopy(port[2]))
                        perm.append(port[0])
                    except StopIteration:
                        new_output.add(output_port)  # treat as closed
                        perm.append(next(element_extra_mode_number))
            else:  # port is closed
                new_output.add(output_port)
                perm.append(next(element_extra_mode_number))

        if position_counter:
            for i in range(position_counter):
                new_output.add(OutputPort())
                new_input.add(InputPort())
                perm.append(next(element_extra_mode_number))

        remaining_ports = sorted(list(element_active_port) + list(element_inactive_port))
        for port in remaining_ports:
            new_input.add(deepcopy(port[1]))
            new_output.add(deepcopy(port[2]))
            perm.append(port[0])

        self._adjust_orderings(perm)
        self._output = new_output
        self.set_children([self._objects, self._output.ports])
        self._input = new_input

    def _adjust_orderings(self, perm: List[int]):
        pad = list(range(self.modes, len(perm)))
        self.permutations = [perm + pad for perm in self.permutations]  # append modes to all components
        self.permutations.append(perm)  # add permutation for most recent component

    @DefaultCache(time_arg=False)
    def times(self, parameters: dict = None):
        return merge_times([merge_times([function.times(parameters) for function in self._objects]),
                            self.output.times(self.set_parameters(parameters))])

    @DefaultCache(time_arg=True)
    def evaluate_quadruple(self, t: float, parameters: Union[dict, frozendict] = None):
        if self._elements:
            return self._rule([component.evaluate_quadruple(t, parameters).match(self.permutations[i])
                               for i, component in enumerate(self._elements)])
        else:
            return EvaluatedQuadruple()

    @DefaultCache(time_arg=True)
    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        return any(function.is_time_dependent(t, parameters) for function in self._objects) \
            or self.output.is_time_dependent(t, parameters)

    @DefaultCache(time_arg=True)
    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None) -> bool:
        return any(element.is_nonhermitian_time_dependent(t, parameters)
                   if not isinstance(element, AScatteringMatrix) else
                   element.is_time_dependent(t, parameters) for element in self.elements.values()) \
            or self.output.is_time_dependent(t, parameters)

    def _cache_clear(self):
        super()._cache_clear()
        self.times.cache_clear()
        self.evaluate_quadruple.cache_clear()
        self.is_time_dependent.cache_clear()
        self.is_nonhermitian_time_dependent.cache_clear()


def make_masked_source(source: AComponent, port: int):
    port = source.unmasked_position(port) if isinstance(source, Component) else port

    masked_source = Component(source, masked=False)
    for i in range(masked_source.modes):
        if i != port:
            masked_source.output.ports[i].close()
    masked_source.mask()

    return masked_source
