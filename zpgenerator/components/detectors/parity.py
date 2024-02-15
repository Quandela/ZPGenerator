from .base_detector import DetectorComponent
from ..losses.uniform import UniformLoss
from ...virtual.configuration import ParityDetectorGate
from typing import Union
from ...time import ATimeFunction, TimeInterval


class ParityDetector(DetectorComponent):
    """ A detector that measures the parity summation of an output """
    def __init__(self, gate: Union[ATimeFunction, TimeInterval, list] = None, efficiency: float = 1,
                 parameters: dict = None, name: str = None, bin_name: str = None):
        super().__init__(parameters=parameters, name=name)
        self.add(UniformLoss(efficiency=efficiency))
        self.add(ParityDetectorGate(efficiency=1, gate=gate, parameters=parameters), bin_name=bin_name)
        self.default_name = '_PD'
