from ...system import NaturalSystem, HamiltonianBase, EnvironmentBase, EmitterBase, LindbladVector
from ...time import Operator
from ...time.parameters import parinit
import numpy as np
import qutip as qt


class BiexcitonSystem(NaturalSystem):
    """
    A four-level 'diamond' shaped system with optional fine structure splitting.
    """

    def __init__(self, parameters: dict = None, name: str = None):
        """
        :param parameters:
        :param name:
        """

        # states
        st = {'|g>': qt.fock(4, 0),
              '|x>': qt.fock(4, 1),
              '|y>': qt.fock(4, 2),
              '|b>': qt.fock(4, 3)}

        # operators
        op = {'lower_x': st['|g>'] * st['|x>'].dag(),
              'lower_y': st['|g>'] * st['|y>'].dag(),
              'lower_bx': st['|x>'] * st['|b>'].dag(),
              'lower_by': st['|y>'] * st['|b>'].dag(),
              'raise_x': st['|x>'] * st['|g>'].dag(),
              'raise_y': st['|y>'] * st['|g>'].dag(),
              'raise_bx': st['|b>'] * st['|x>'].dag(),
              'raise_by': st['|b>'] * st['|y>'].dag(),
              'population_x': st['|x>'] * st['|x>'].dag(),
              'population_y': st['|y>'] * st['|y>'].dag(),
              'population_b': st['|b>'] * st['|b>'].dag(),
              'lower_fss': st['|x>'] * st['|y>'].dag(),
              'Z_fss': st['|y>'] * st['|y>'].dag() - st['|x>'] * st['|x>'].dag()}

        # HamiltonianBase definition
        ham = HamiltonianBase()

        def resonance(args: dict):
            return args['resonance'] * (op['population_x'] + op['population_y'])

        ham.add(Operator(resonance, parameters=parinit({'resonance': 0}, parameters)))

        def fss(args: dict):
            return args['fss'] * op['Z_fss'] / 2

        ham.add(Operator(fss, parameters=parinit({'fss': 0}, parameters)))

        def binding(args: dict):
            return (-args['binding'] + 2 * args['resonance']) * op['population_b']

        ham.add(Operator(binding, parameters=parinit({'resonance': 0, 'binding': 100}, parameters)))

        # EnvironmentBase definition
        env = EnvironmentBase()

        def collapseX(args: dict):
            return np.sqrt(args['decay']) * (np.cos(args['theta_c']) * op['lower_x'] +
                                             np.sin(args['theta_c']) * np.exp(1.j * args['phi_c']) * op['lower_y'])

        def collapseY(args: dict):
            return np.sqrt(args['decay']) * (np.sin(args['theta_c']) * op['lower_x'] -
                                             np.cos(args['theta_c']) * np.exp(1.j * args['phi_c']) * op['lower_y'])

        def collapseBX(args: dict):
            return np.sqrt(args['decay_b']) * (np.cos(args['theta_bc']) * op['lower_bx'] +
                                               np.sin(args['theta_bc']) * np.exp(1.j * args['phi_bc']) * op['lower_by'])

        def collapseBY(args: dict):
            return np.sqrt(args['decay_b']) * (np.sin(args['theta_bc']) * op['lower_bx'] -
                                               np.cos(args['theta_bc']) * np.exp(1.j * args['phi_bc']) * op['lower_by'])

        env.add(Operator(collapseX, parameters=parinit({'decay': 1, 'theta_c': 0, 'phi_c': 0}, parameters)))
        env.add(Operator(collapseY, parameters=parinit({'decay': 1, 'theta_c': 0, 'phi_c': 0}, parameters)))
        env.add(Operator(collapseBX, parameters=parinit({'decay_b': 2, 'theta_bc': 0, 'phi_bc': 0}, parameters)))
        env.add(Operator(collapseBY, parameters=parinit({'decay_b': 2, 'theta_bc': 0, 'phi_bc': 0}, parameters)))

        # Other environmental effects

        def emitter_dephasing(args: dict):
            return np.sqrt(args['dephasing']) * (op['population_x'] + op['population_y'] + 2 * op['population_b'])

        env.add(Operator(emitter_dephasing, parameters=parinit({'dephasing': 0}, parameters)))

        def fss_dephasing(args: dict):
            return np.sqrt(args['dephasing_fss']) * op['Z_fss'] / 2

        env.add(Operator(fss_dephasing, parameters=parinit({'dephasing_fss': 0}, parameters)))

        super().__init__(hamiltonian=ham, environment=env, states=st, operators=op, name=name)


class BiexcitonEmitter(EmitterBase):
    """
    A four-level double cascaded emitter.
    """

    def __init__(self, parameters: dict = None, name: str = None):
        system = BiexcitonSystem(parameters, name)

        # take first four environmental operators as collapse operators
        transitions = LindbladVector(operators=system.environment.operator_list[0:4], parameters=parameters)

        super().__init__()
        self.set_system(system=system, transitions=transitions)
