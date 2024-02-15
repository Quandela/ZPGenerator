from ...system import NaturalSystem, HamiltonianBase, EnvironmentBase, EmitterBase, LindbladVector
from ...time import Operator
from ...time.parameters import parinit
import numpy as np
import qutip as qt


class ExcitonSystem(NaturalSystem):
    """
    A three-level V-shape system with optional fine structure splitting.
    """

    def __init__(self, parameters: dict = None, name: str = None):
        """
        :param parameters:
        :param name:
        """

        # states
        st = {'|g>': qt.fock(3, 0),
              '|x>': qt.fock(3, 1),
              '|y>': qt.fock(3, 2)}

        # operators
        op = {'lower_x': st['|g>'] * st['|x>'].dag(),
              'lower_y': st['|g>'] * st['|y>'].dag(),
              'raise_x': st['|x>'] * st['|g>'].dag(),
              'raise_y': st['|y>'] * st['|g>'].dag(),
              'population_x': st['|x>'] * st['|x>'].dag(),
              'population_y': st['|y>'] * st['|y>'].dag(),
              'lower_fss': st['|x>'] * st['|y>'].dag(),
              'Z_fss': st['|y>'] * st['|y>'].dag() - st['|x>'] * st['|x>'].dag()}

        # HamiltonianBase definition
        ham = HamiltonianBase()

        def detuning(args: dict):
            return args['resonance'] * (op['population_x'] + op['population_y'])

        ham.add(Operator(detuning, parameters=parinit({'resonance': 0}, parameters)))

        def fss(args: dict):
            return args['fss'] * op['Z_fss'] / 2

        ham.add(Operator(fss, parameters=parinit({'fss': 0}, parameters)))

        env = EnvironmentBase()

        def collapseX(args: dict):
            return np.sqrt(args['decay']) * (np.cos(args['theta_c']) * op['lower_x'] +
                                             np.sin(args['theta_c']) * np.exp(1.j * args['phi_c']) * op['lower_y'])

        def collapseY(args: dict):
            return np.sqrt(args['decay']) * (np.sin(args['theta_c']) * op['lower_x'] -
                                             np.cos(args['theta_c']) * np.exp(1.j * args['phi_c']) * op['lower_y'])

        env.add(Operator(collapseX, parameters=parinit({'decay': 1, 'theta_c': 0, 'phi_c': 0}, parameters)))
        env.add(Operator(collapseY, parameters=parinit({'decay': 1, 'theta_c': 0, 'phi_c': 0}, parameters)))

        def emitter_dephasing(args: dict):
            return np.sqrt(args['dephasing']) * (op['population_x'] + op['population_y'])

        env.add(Operator(emitter_dephasing, parameters=parinit({'dephasing': 0}, parameters)))

        def fss_dephasing(args: dict):
            return np.sqrt(args['dephasing_fss']) * op['Z_fss'] / 2

        env.add(Operator(fss_dephasing, parameters=parinit({'dephasing_fss': 0}, parameters)))

        # add polarization flips?

        super().__init__(hamiltonian=ham, environment=env, states=st, operators=op, parameters=parameters, name=name)


class ExcitonEmitter(EmitterBase):
    """
    A three-level V-shape system with optional fine structure splitting.
    """

    def __init__(self, parameters: dict = None, name: str = None):
        """
        :param parameters:
        :param name:
        """
        system = ExcitonSystem(parameters, name)

        transitions = LindbladVector(operators=system.environment.operator_list[0:2], parameters=parameters)

        super().__init__()
        self.set_system(system=system, transitions=transitions)
