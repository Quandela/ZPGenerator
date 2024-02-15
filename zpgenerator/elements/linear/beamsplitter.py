from ...system import ScattererBase
from ...time.parameters import parinit
from numpy import pi, cos, sin
from qutip import Qobj


class BeamSplitter(ScattererBase):
    """
    A beam splitter linear-optical component
    """

    def __init__(self,
                 parameters: dict = None,
                 name: str = None):
        """
        :param parameters: a list of parameters that will set the default parameters for the system.
        :param name: the optional name of the system to distinguish it from other similar nonlinear.
        """
        # Default parameters
        parameters = parinit({'angle': pi / 4}, parameters)

        def unitary(args: dict):
            return Qobj([[cos(args['angle']), 1.j * sin(args['angle'])],
                         [1.j * sin(args['angle']), cos(args['angle'])]])

        super().__init__(unitary, parameters=parameters, name=name)
        self.default_name = '_BS'