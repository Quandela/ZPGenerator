from .base_detector import DetectorComponent
from ..circuit import Circuit
from ...network.detector import DetectorGate
from typing import Union, List


class TimePartitionDetector(DetectorComponent):

    def __init__(self, thresholds: Union[list, List[list]], resolution: int = None, efficiency: int = 1,
                 parameters: dict = None, name: str = None, bin_name: str = None):
        super().__init__(Circuit.loss(efficiency), parameters=parameters, name=name)
        self.default_name = '_TPD'
        time_bin_number = len(thresholds)

        if not isinstance(thresholds[0], list):
            thresholds = [[thresholds[i], thresholds[i + 1] if i + 1 < time_bin_number else float('inf')]
                          for i in range(time_bin_number)]

        for i in range(time_bin_number):
            detector = DetectorGate(resolution=1 if resolution is None else resolution,
                                    gate=thresholds[i], parameters=parameters,
                                    method='Threshold' if resolution is None else 'Fourier')
            self.add(0, detector, bin_name=bin_name if bin_name else name + ' bin ' + str(i) if name else None)
