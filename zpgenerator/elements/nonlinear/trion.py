from ...time import Operator
from ...time.parameters import parinit
from ...system import NaturalSystem, HamiltonianBase, EnvironmentBase, EmitterBase, LindbladVector
import qutip as qt
import numpy as np


class TrionSystem(NaturalSystem):
    """ A four-level emitter (qubit * exciton) with optional dephasing and resonance shift."""
    def __init__(self,
                 charge: str = 'negative',
                 parameters: dict = None,
                 name: str = None):
        """
        :param charge: whether the trion has a negative or positive charge.
        :param parameters: a list of parameters that will set the default parameters for the system.
        :param name: the optional name of the system to distinguish it from other similar nonlinear.
        """

        #  Timescale (picoseconds) for magnetic field parameters in Tesla
        # mu = 9.274 * 1e-24  # units of Joule/Tesla
        # hbar = 1.0545 * 1e-34  # units of Joule * Second
        # picosecond = 1e-12  # seconds
        # coef = mu / hbar * picosecond

        #  states
        st = {'|spin_up>': qt.fock(4, 0),
              '|spin_down>': qt.fock(4, 1),
              '|trion_up>': qt.fock(4, 2),
              '|trion_down>': qt.fock(4, 3)}

        # changing 4-dimensional system into a 2x2 system
        for s in st.values():
            s.dims = [[2, 2], [1, 1]]

        #  operators
        op = {'lower_R': st['|spin_up>'] * st['|trion_up>'].dag(),
              'lower_L': st['|spin_down>'] * st['|trion_down>'].dag(),
              'raise_R': st['|trion_up>'] * st['|spin_up>'].dag(),
              'raise_L': st['|trion_down>'] * st['|spin_down>'].dag(),
              'population_R': st['|trion_up>'] * st['|trion_up>'].dag(),
              'population_L': st['|trion_down>'] * st['|trion_down>'].dag(),
              'X': st['|spin_up>'] * st['|spin_down>'].dag() +
                   st['|spin_down>'] * st['|spin_up>'].dag(),
              'Y': -1.j * st['|spin_up>'] * st['|spin_down>'].dag() +
                   1.j * st['|spin_down>'] * st['|spin_up>'].dag(),
              'Z': st['|spin_up>'] * st['|spin_up>'].dag() -
                   st['|spin_down>'] * st['|spin_down>'].dag(),
              'X_trion': st['|trion_up>'] * st['|trion_down>'].dag() +
                         st['|trion_down>'] * st['|trion_up>'].dag(),
              'Y_trion': -1.j * st['|trion_up>'] * st['|trion_down>'].dag() +
                         1.j * st['|trion_down>'] * st['|trion_up>'].dag(),
              'Z_trion': st['|trion_up>'] * st['|trion_up>'].dag() -
                         st['|trion_down>'] * st['|trion_down>'].dag()}

        # HamiltonianBase definition
        ham = HamiltonianBase()

        def resonance(args: dict):
            return args['resonance'] * (op['population_L'] + op['population_R'])

        ham.add(Operator(resonance, parameters=parinit({'resonance': 0}, parameters)))

        def spin(args: dict):
            return (args['g_spin'] / 2) * (args['Bx'] * op['X'] + args['By'] * op['Y'] +
                                           args['Bz'] * op['Z']) + \
                (args['g_trion'] / 2) * (args['Bx'] * op['X_trion'] + args['By'] * op['Y_trion'] +
                                         args['Bz'] * op['Z_trion'])

        ham.add(Operator(spin, parameters=parinit({'Bx': 0, 'By': 0, 'Bz': 0, 'g_spin': 2, 'g_trion': 2}, parameters)))

        opx = op['X'] if charge == 'negative' else op['X_trion']
        opy = op['Y'] if charge == 'negative' else op['Y_trion']
        opz = op['Z'] if charge == 'negative' else op['Z_trion']

        def overhauser(args: dict):
            return (args['g_spin'] / 2) * (args['Bx_OH'] * opx + args['By_OH'] * opy + args['Bz_OH'] * opz)

        ham.add(Operator(overhauser, parameters=parinit({'Bx_OH': 0, 'By_OH': 0, 'Bz_OH': 0, 'g_spin': 2}, parameters)))

        # EnvironmentBase definition
        env = EnvironmentBase()

        opH = (op['lower_R'] + op['lower_L']) / np.sqrt(2)  # R = H - iV, L = H + iV, H = R + L, V = (L - R)/i
        opV = 1.j * (op['lower_R'] - op['lower_L']) / np.sqrt(2)

        def collapse0(args: dict):
            return np.sqrt(args['decay']) * (np.cos(args['theta_c']) * opH +
                                             np.sin(args['theta_c']) * opV * np.exp(1.j * args['phi_c']))

        def collapse1(args: dict):
            return np.sqrt(args['decay']) * (np.sin(args['theta_c']) * opH -
                                             np.cos(args['theta_c']) * opV * np.exp(1.j * args['phi_c']))

        env.add(
            [Operator(collapse0, parameters=parinit({'decay': 1, 'theta_c': np.pi / 4, 'phi_c': -np.pi / 2}, parameters))])
        env.add(
            [Operator(collapse1, parameters=parinit({'decay': 1, 'theta_c': np.pi / 4, 'phi_c': -np.pi / 2}, parameters))])

        def emitter_dephasing(args: dict):
            return np.sqrt(args['dephasing']) * (op['population_L'] + op['population_R'])

        env.add(Operator(emitter_dephasing, parameters=parinit({'dephasing': 0}, parameters)))

        def spin_dephasing(args: dict):
            return np.sqrt(args['dephasing_spin']) * op['Z'] / 2

        env.add(Operator(spin_dephasing, parameters=parinit({'dephasing_spin': 0}, parameters)))

        def trion_dephasing(args: dict):
            return np.sqrt(args['dephasing_trion']) * op['Z_trion'] / 2

        env.add(Operator(trion_dephasing, parameters=parinit({'dephasing_trion': 0}, parameters)))

        # we should add spin flips here eventually

        super().__init__(hamiltonian=ham, environment=env, states=st, operators=op, parameters=parameters, name=name)


class TrionEmitter(EmitterBase):
    """
    A four-level trion emitter.
    """

    def __init__(self,
                 charge: str = 'negative',
                 parameters: dict = None,
                 name: str = None):
        system = TrionSystem(charge, parameters, name)

        transitions = LindbladVector(operators=system.environment.operator_list[0:2], parameters=parameters)

        super().__init__()
        self.set_system(system=system, transitions=transitions)
