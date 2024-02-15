from .base_detector import DetectorComponent
from ..losses.uniform import UniformLoss
from ...network.detector import DetectorGate
from ...time import ATimeFunction, TimeInterval
from typing import Union


class VacuumDetector(DetectorComponent):
    """ A detector that returns the probability of vacuum """

    def __init__(self, gate: Union[ATimeFunction, TimeInterval, list] = None, efficiency: float = 1,
                 parameters: dict = None, name: str = None, bin_name: str = None):
        super().__init__(parameters=parameters, name=name)
        self.add(UniformLoss(efficiency=efficiency))
        self.add(DetectorGate(resolution=0, gate=gate, parameters=parameters, method='Threshold'), bin_name=bin_name)
        self.default_name = '_VD'
