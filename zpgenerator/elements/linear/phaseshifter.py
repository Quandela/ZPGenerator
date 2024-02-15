from ...system import ScattererBase
from ...time.parameters import parinit
from qutip import Qobj
from numpy import pi, exp


class PhaseShifter(ScattererBase):
    """
    A phase shift on a single mode
    """

    def __init__(self,
                 parameters: dict = None,
                 name: str = None):
        """
        :param parameters: a list of parameters that will set the default parameters for the system.
        :param name: the optional name of the system to distinguish it from other similar nonlinear.
        """
        # Default parameters
        parameters = parinit({'phase': pi}, parameters)

        def unitary(args: dict):
            return Qobj([[exp(1.j * args['phase'])]])

        super().__init__(unitary, parameters=parameters, name=name)
        self.default_name = '_PS'
