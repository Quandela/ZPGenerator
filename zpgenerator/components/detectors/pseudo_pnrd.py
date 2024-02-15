# from .base_detector import DetectorComponent
# from ..loss.uniform import UniformLoss
# from ...network.detector import DetectorGate
# from ...time import ATimeFunction, TimeInterval
# from typing import Union
#
#
# class PseudoPNRDetector(DetectorComponent):
#     """ A series of beam splitters followed by a threshold detector """
#
#     def __init__(self, resolution: int, gate: Union[ATimeFunction, TimeInterval, list], efficiency: float = 1,
#                  parameters: dict = None, name: str = None):
#         elements = [UniformLoss(efficiency=efficiency),
#                     BeamSplitterCascade(number=resolution)]
#
#         super().__init__(elements, parameters=parameters, name=name)
#
#         for i in range(self.modes):
#             self.add(i, DetectorGate(resolution=1, gate=gate, parameters=parameters))
