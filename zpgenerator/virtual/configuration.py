from abc import abstractmethod
from ..network.detector import ADetectorGate, DetectorGate
from ..time import TimeInterval, TimeIntervalFunction
from typing import Union, List
import numpy as np


class AVirtualDetectorGate(ADetectorGate):
    """
    A detector that provides a set of virtual configurations and inverse transform used for solving the ZPG.
    """

    @property
    @abstractmethod
    def virtual_configurations(self) -> List[complex]:
        pass

    @property
    @abstractmethod
    def inverse_transform(self) -> str:
        pass


class ParityDetectorGate(DetectorGate, AVirtualDetectorGate):
    """
    A detector with virtual configuration of 2
    """

    def __init__(self,
                 efficiency: Union[callable, int, float] = 1.,
                 gate: Union[TimeIntervalFunction, TimeInterval, list] = None,
                 parameters: dict = None,
                 name: str = None):
        """
        :param efficiency: a constant physical efficiency multiplier for the detector.
        :param gate: an optional function or interval describing the detector gate and efficiency.
        """
        super().__init__(resolution=0, efficiency=efficiency, gate=gate, parameters=parameters, name=name)

    @property
    def virtual_configurations(self) -> List[complex]:
        return [2]

    @property
    def inverse_transform(self) -> str:
        return 'Identity'


class PhysicalDetectorGate(DetectorGate, AVirtualDetectorGate):
    """
    A detector with physical efficiency configurations (those bounded between 0 and 1).
    """

    def __init__(self,
                 resolution: int = 1,
                 efficiency: Union[callable, int, float] = 1.,
                 gate: Union[TimeIntervalFunction, TimeInterval, list] = None,
                 ignore_zero: bool = False,
                 parameters: dict = None,
                 name: str = None):
        """
        :param resolution: an integer describing the number of configurations.
        :param efficiency: a constant physical efficiency multiplier for the detector.
        :param gate: an optional function or interval describing the detector gate and efficiency.
        :param ignore_zero: an option to neglect the zero-efficiency configuration.
        """
        super().__init__(resolution=resolution, efficiency=efficiency, gate=gate, parameters=parameters, name=name)
        self.ignore_zero = ignore_zero
        self.virtual_configurations = self.resolution

    @property
    def virtual_configurations(self) -> List[complex]:
        return self._virtual_configurations

    @virtual_configurations.setter
    def virtual_configurations(self, resolution):
        self._virtual_configurations = list(reversed([n / resolution
                                                      for n in range(1 if self.ignore_zero else 0, resolution + 1)]
                                                     )) if resolution != 0 else [0.0]
        self._resolution = resolution - 1 if self.ignore_zero else resolution

    @property
    def inverse_transform(self) -> str:
        return 'Linear Inverse'


class FourierDetectorGate(DetectorGate, AVirtualDetectorGate):
    """
    A detector with virtual configurations that are roots of unity.
    """

    def __init__(self,
                 resolution: int = 1,
                 efficiency: Union[callable, int, float] = 1.,
                 gate: Union[TimeIntervalFunction, TimeInterval, list] = None,
                 parameters: dict = None,
                 name: str = None):
        """
        :param resolution: an integer describing the number of configurations.
        :param efficiency: a constant physical efficiency multiplier for the detector.
        :param gate: an optional function or interval describing the detector gate and efficiency.
        """
        super().__init__(resolution=resolution, efficiency=efficiency, gate=gate, parameters=parameters, name=name)
        self.virtual_configurations = self.resolution

    @property
    def virtual_configurations(self) -> List[complex]:
        return self._virtual_configurations

    @virtual_configurations.setter
    def virtual_configurations(self, resolution):
        self._virtual_configurations = [1 - np.exp(-1.j * 2 * np.pi * n / (resolution + 1))
                                        for n in range(0, resolution + 1)]
        self._resolution = resolution

    @property
    def inverse_transform(self) -> str:
        return 'Fourier Transform'
