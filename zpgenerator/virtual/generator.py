from ..system import AElement
from ..network import Component, AComponent
from .propagator import VPropNHTD, VPropHTD, VPropTI
from qutip import Options


class Generator:
    """a propagator factory that chooses the right propagator for a given time step"""

    def __init__(self, component: AElement, binned_detectors: dict = None,
                 lifetime_mode: int = None, precision: int = 6):
        self.component = component if isinstance(component, AComponent) else Component(component)
        self.default_options = Options(nsteps=500000, atol=10 ** -precision, rtol=10 ** -(precision))
        self.lifetime_mode = lifetime_mode
        self.binned_detectors = self.component.output.binned_detectors if binned_detectors is None else binned_detectors

    def build_propagator(self, t: float, parameters: dict = None, options: Options = None):
        options = self.default_options if options is None else options

        quadruple = self.component.evaluate_quadruple(t, parameters)
        hamiltonian = quadruple.hamiltonian
        environment = quadruple.environment
        transitions = quadruple.transitions

        population = transitions[self.lifetime_mode].num() if self.lifetime_mode is not None else None
        if population:
            expect_operator = [population.constant] if not population.variable \
                else [lambda t, rho_t: (population.evaluate(t) * rho_t).tr()]
        else:
            expect_operator = None

        jumps = [sum(time_bin.detector.coupling_function(t, self.component.set_parameters(parameters)) *
                     transitions[time_bin.mode].jump()
                     for time_bin in time_bins) for time_bins in self.binned_detectors.values()]

        if self.component.is_time_dependent(t, parameters) or population is not None:
            if self.component.is_nonhermitian_time_dependent(t, parameters):
                generator = hamiltonian.liou() + sum(env.lind() if not env.is_super else env for env in environment)
                return VPropNHTD(generator=generator,
                                 jumps=jumps,
                                 expect_operators=expect_operator,
                                 options=options)
            else:
                return VPropHTD(hamiltonian=hamiltonian.list_form(),
                                collapse_operators=[env.list_form() for env in environment],
                                jumps=[jump.constant for jump in jumps],
                                expect_operators=expect_operator,
                                options=options)
        else:
            #  Add TI method eventually
            return VPropHTD(hamiltonian=hamiltonian.list_form(),
                            collapse_operators=[env.list_form() for env in environment],
                            jumps=[jump.constant for jump in jumps],
                            expect_operators=expect_operator,
                            options=options)
