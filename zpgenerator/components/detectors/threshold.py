from .base_detector import DetectorComponent
from ..losses.uniform import UniformLoss
from ...network.detector import DetectorGate
from ...time import ATimeFunction, TimeInterval
from typing import Union


class ThresholdDetector(DetectorComponent):
    """ A threshold detector """

    def __init__(self, gate: Union[ATimeFunction, TimeInterval, list] = None, efficiency: float = 1,
                 parameters: dict = None, name: str = None, bin_name: str = None):
        super().__init__(parameters=parameters, name=name)
        self.add(UniformLoss(efficiency=efficiency))
        self.add(DetectorGate(resolution=1, gate=gate, parameters=parameters), bin_name=bin_name)
        self.default_name = '_TD'
