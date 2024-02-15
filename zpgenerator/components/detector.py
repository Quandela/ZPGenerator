from .detectors import *
from ..time import ATimeFunction, TimeInterval
from typing import Union, List


class Detector(DetectorComponent):
    """ A detector factory """

    @classmethod
    def pnr(cls, resolution: int, gate: Union[ATimeFunction, TimeInterval, list] = None, efficiency: float = 1,
            parameters: dict = None, name: str = None, bin_name: str = None):
        return PNRDetector(resolution=resolution, gate=gate, efficiency=efficiency, parameters=parameters, name=name,
                           bin_name=bin_name)

    @classmethod
    def threshold(cls, gate: Union[ATimeFunction, TimeInterval, list] = None, efficiency: float = 1,
                  parameters: dict = None, name: str = None, bin_name: str = None):
        return ThresholdDetector(gate=gate, efficiency=efficiency, parameters=parameters, name=name, bin_name=bin_name)

    @classmethod
    def vacuum(cls, gate: Union[ATimeFunction, TimeInterval, list] = None, efficiency: float = 1,
                  parameters: dict = None, name: str = None, bin_name: str = None):
        return VacuumDetector(gate=gate, efficiency=efficiency, parameters=parameters, name=name, bin_name=bin_name)


    @classmethod
    def parity(cls, gate: Union[ATimeFunction, TimeInterval, list] = None, efficiency: float = 1,
               parameters: dict = None, name: str = None, bin_name: str = None):
        return ParityDetector(gate=gate, efficiency=efficiency, parameters=parameters, name=name, bin_name=bin_name)

    @classmethod
    def partition(cls, thresholds: Union[list, List[list]], resolution: int = 1, efficiency: int = 1,
                  parameters: dict = None, name: str = None, bin_name: str = None):
        return TimePartitionDetector(thresholds=thresholds, resolution=resolution, efficiency=efficiency,
                                     parameters=parameters, name=name, bin_name=bin_name)
