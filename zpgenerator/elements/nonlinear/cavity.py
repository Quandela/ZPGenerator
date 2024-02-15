from ...system import NaturalSystem, HamiltonianBase, EnvironmentBase, EmitterBase, LindbladVector
from ...time import Operator, TimeOperator, TimeIntervalFunction, CompositeTimeOperator, PulseBase, Lifetime, \
    TimeInterval
from ...time.parameters import parinit
from numpy import linspace, sqrt, exp, pad, array
from qutip import fock, destroy
from typing import Union
from scipy.interpolate import interp1d
from scipy.integrate import simpson, cumulative_trapezoid


class CavitySystem(NaturalSystem):
    """
    A cavity (stationary harmonic oscillator).
    """

    def __init__(self,
                 truncation: int = 2,
                 modes: int = 1,
                 decay: bool = True,
                 parameters: dict = None,
                 name: str = None):
        """
        :param truncation: the number of energy levels considered.
        :param modes: the number of emission modes collected with equal weight.
        :param parameters: a list of parameters that will set the default parameters for the system.
        :param name: the optional name of the system to distinguish it from other similar nonlinear.
        """

        # states
        st = dict()
        for i in range(0, truncation):
            st.setdefault('|' + str(i) + '>', fock(truncation, i))

        # operators
        d = destroy(truncation)

        op = {'annihilation': d,
              'number': d.dag() * d,
              'position': (d.dag() + d) / sqrt(2),
              'momentum': (d.dag() - d) / sqrt(2)}

        # HamiltonianBase definition
        ham = HamiltonianBase()

        def resonance(parameters: dict):
            return parameters['resonance'] * op['number']

        ham.add(Operator(matrix=resonance, parameters=parinit({'resonance': 0.}, parameters)))

        # EnvironmentBase definition
        env = EnvironmentBase()

        def dephasing(args: dict):
            return sqrt(args['dephasing']) * op['number']

        env.add(Operator(matrix=dephasing, parameters=parinit({'dephasing': 0.}, parameters)))

        if decay:
            def collapse(parameters: dict):
                return sqrt(parameters['decay'] / modes) * op['annihilation']

            env.add([Operator(matrix=collapse, parameters=parinit({'decay': 1.}, parameters))] * modes)

        super().__init__(hamiltonian=ham,
                         environment=env,
                         states=st,
                         operators=op,
                         parameters=parameters,
                         name=name)


class ShapedCavitySystem(CavitySystem):
    """
    A cavity with a time-dynamic emission rate to shape emission
    """

    def __init__(self,
                 shape: Union[PulseBase, Lifetime] = None,
                 resolution: int = 600,
                 truncation: int = 2,
                 parameters: dict = None,
                 name: str = None):

        if shape is None:
            self.shape_function = lambda t, args: args['decay'] * exp(-(t - args['delay']) * args['decay']) \
                if (t - args['delay']) >= 0 else 0
            self.decay_function = lambda t, args: args['decay']
            self.interval = TimeInterval(interval=lambda args: [args['delay'], args['delay'] + 10 / args['decay']],
                                         parameters={'delay': 0., 'decay': 1} | (parameters if parameters else {}),
                                         name=name)
            super().__init__(truncation=truncation, modes=1, decay=False, parameters=parameters, name=name)

            def collapse(parameters: dict):
                return sqrt(parameters['decay']) * self.operators['annihilation']

            interval = TimeInterval(['delay', float('inf')], parameters=parinit({'delay': 0.}, parameters))

            collapse = TimeOperator(operator=collapse,
                                    functions=TimeIntervalFunction(value=1, interval=interval),
                                    parameters=parinit({'decay': 1.}, parameters))

            self.environment.add(CompositeTimeOperator(operators=[collapse]))

        else:
            if isinstance(shape, PulseBase):
                times = shape.times(parameters)  # forces default parameters!
                times = linspace(times[0], times[-1], resolution)
                shape = [shape.evaluate(t, parameters) for t in times]
            elif isinstance(shape, Lifetime):
                times = shape.times
                shape = interp1d(times, shape.population)
                times = linspace(times[0], times[-1], resolution)
                shape = shape(times)
            else:
                assert False, "Cannot create oscillator with the requested pulse shape."

            self.interval = [times[0], times[-1]]

            # normalizing the shape
            norm = simpson(shape, times)
            shape = array(shape) / norm

            # determine the appropriate time-dependent decay rate:
            # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.123604
            decay = pad(cumulative_trapezoid(shape, times), (1, 0), 'constant')
            decay = shape / (1 - decay)
            decay = interp1d(times, decay)

            def decay_function(t, args):
                try:
                    return abs(decay(t))
                except ValueError:
                    return 0

            shape = interp1d(times, shape)

            def shape_function(t, args):
                try:
                    return shape(t)
                except ValueError:
                    return 0

            self.shape_function = shape_function
            self.decay_function = decay_function

            super().__init__(truncation=truncation, modes=1, decay=False, parameters=parameters, name=name)

            collapse = TimeOperator(operator=destroy(truncation),
                                    functions=TimeIntervalFunction(
                                        value=lambda t, args: sqrt(self.decay_function(t, args)),
                                        interval=self.interval))

            self.environment.add(CompositeTimeOperator(operators=[collapse]))


class CavityEmitter(EmitterBase):
    """
    A cavity emitter with exponential decay
    """

    def __init__(self,
                 truncation: int = 2,
                 modes: int = 1,
                 parameters: dict = None,
                 name: str = None):
        system = CavitySystem(truncation=truncation, modes=modes, parameters=parameters, name=name)

        transitions = LindbladVector(operators=system.environment.operator_list[1:modes + 1], parameters=parameters)

        super().__init__()
        self.set_system(system=system, transitions=transitions)


class ShapedCavityEmitter(EmitterBase):
    """
    A cavity emitter with shaped emission
    """

    def __init__(self,
                 shape: Union[PulseBase, Lifetime] = None,
                 resolution: int = 600,
                 truncation: int = 2,
                 parameters: dict = None,
                 name: str = None):
        system = ShapedCavitySystem(shape=shape, resolution=resolution, truncation=truncation,
                                    parameters=parameters, name=name)

        transitions = LindbladVector(operators=system.environment.operator_list[1], parameters=parameters)

        super().__init__()
        self.set_system(system=system, transitions=transitions)
