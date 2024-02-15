from .quality import ProcessorQuality
from .algorithms.distributions import CorrelationDistribution, StateDistribution, ChannelDistribution
from ..misc.display import Display
from ..network import AComponent, ADetectorGate, Component
from ..system import AElement
from typing import Union, List
from qutip import Options, Qobj


class Processor(ProcessorQuality):
    """Front end Processor class"""

    def __init__(self, component: Union[AElement, AComponent] = None):
        """
        Create an empty processor to add components later, or initialize it with a component.
        :param component: A source, circuit, or detector component.
        """
        super().__init__(component)

    @property
    def bin_labels(self):
        """
        :return: a list of labels corresponding to detection bins
        """
        return list(self.component.output.binned_detectors.keys())

    @property
    def bins(self):
        """
        :return: the number of detection bins
        """
        return len(self.bin_labels)

    def add(self, position: Union[int, List[int]], element: Union[AElement, ADetectorGate],
            parameters: dict = None, name: str = None, bin_name: str = None):
        if isinstance(position, list):
            for i in position:
                super().add(i, element, parameters, name, bin_name)
        else:
            super().add(position, element, parameters, name, bin_name)

    def display(self, elements=False):
        if elements:
            for i, element in enumerate(self.component.elements.values()):
                modes = element.modes
                perm = self.component.permutations[i]
                ports = [j for j, k in enumerate(perm) if k < modes]
                Display(element, ports).display()
        else:
            comp = Component(self.component)
            for port in comp.input.ports:
                port.close()
            Display(comp).display()

    def probs(self, parameters: dict = None, bin_list: list = None, chop: bool = True,
              options: Options = None, reset: bool = True):
        probs = CorrelationDistribution(super().probs(parameters=parameters, bin_list=bin_list,
                                                      options=options, reset=reset),
                                        precision=self.precision,
                                        type='real' if self._contains_unnormalised_detector else 'positive')
        if chop:
            probs.chop(normalize=not self._contains_unnormalised_detector and self.initial_state.norm() == 1)
        return probs

    def conditional_states(self, parameters: dict = None, bin_list: list = None,
                           dims: List[int] = None, select: List[int] = None,
                           chop: bool = True, options: Options = None, reset: bool = True):
        states = StateDistribution(super().conditional_states(parameters=parameters, bin_list=bin_list,
                                                              dims=dims, select=select, options=options, reset=reset),
                                   precision=self.precision)
        if chop:
            states.chop(normalize=not self._contains_unnormalised_detector and self.initial_state.norm() == 1)
        return states

    def conditional_channels(self, parameters: dict = None, bin_list: list = None,
                             dims: List[int] = None, select: List[int] = None, basis: List[Qobj] = None,
                             options: Options = None, reset: bool = True):
        return ChannelDistribution(super().conditional_channels(parameters=parameters, bin_list=bin_list,
                                                                dims=dims, select=select, basis=basis,
                                                                options=options, reset=reset),
                                   precision=self.precision)
