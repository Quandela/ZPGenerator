from ...system import NaturalSystem, HamiltonianBase, EnvironmentBase, EmitterBase, LindbladVector
from ...time import Operator
from ...time.parameters import parinit
from numpy import sqrt
import qutip as qt


class QubitSystem(NaturalSystem):
    """
    A two-level system with optional dephasing and detuning.
    """

    def __init__(self,
                 parameters: dict = None,
                 name: str = None,
                 modes: int = 1):
        """
        :param parameters: a list of parameters that will set the default parameters for the system.
        :param name: the optional name of the system to distinguish it and its parameters from other similar nonlinear.
        :param modes: the number of emission modes collected with equal weight.
        """

        # states
        st = {'|g>': qt.fock(2, 0), '|e>': qt.fock(2, 1)}

        # operators
        op = {'lower': qt.destroy(2),
              'number': qt.num(2),
              'X': (qt.destroy(2) + qt.create(2)),
              'Y': (1.j * qt.destroy(2) - 1.j * qt.create(2)),
              'Z': (qt.num(2) - qt.destroy(2) * qt.create(2))}

        # HamiltonianBase definition
        ham = HamiltonianBase()

        def resonance(parameters: dict):
            return parameters['resonance'] * qt.num(2)

        ham.add(Operator(matrix=resonance, parameters=parinit({'resonance': 0.}, parameters)))

        # EnvironmentBase definition
        env = EnvironmentBase()

        def collapse(parameters: dict):
            return sqrt(parameters['decay'] / modes) * qt.destroy(2)

        env.add([Operator(matrix=collapse, parameters=parinit({'decay': 1.}, parameters))] * modes)

        def dephasing(parameters: dict):
            return sqrt(parameters['dephasing']) * qt.num(2)

        env.add(Operator(matrix=dephasing, parameters=parinit({'dephasing': 0.}, parameters)))

        super().__init__(hamiltonian=ham, environment=env, states=st, operators=op, parameters=parameters, name=name)


class TwoLevelEmitter(EmitterBase):
    """
    A two-level system with optional dephasing and resonance shift.
    """

    def __init__(self,
                 parameters: dict = None,
                 name: str = None,
                 modes: int = 1):
        system = QubitSystem(parameters, name, modes)

        transitions = LindbladVector(operators=system.environment.objects[0:modes], parameters=parameters)

        super().__init__()
        self.set_system(system=system, transitions=transitions)
