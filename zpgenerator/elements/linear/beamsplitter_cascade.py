# from ...system import ScattererBase
# from ...time import parinit
# from numpy import pi, cos, sin, sqrt
# from qutip import Qobj
#
#
# class BeamSplitterCascade(ScattererBase):
#     """
#     A cascaded series of beam splitters taking one input and producing 'number' outputs.
#     """
#
#     def __init__(self,
#                  number: int,
#                  parameters: dict = None,
#                  name: str = None):
#         """
#         :param number: the number of outputs
#         :param parameters: a list of parameters that will set the default parameters for the system.
#         :param name: the optional name of the system to distinguish it from other similar nonlinear.
#         """
#
#         super().__init__(unitary, parameters=parameters, name=name)
#
