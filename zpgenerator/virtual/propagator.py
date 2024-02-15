from .state import VState
from ..time import EvaluatedOperator
from abc import ABC, abstractmethod
from qutip import Qobj, Options, mesolve, spre, liouvillian
from typing import Union


class AVirtualPropagator(ABC):
    """
    An object that propagates a virtual state to time t conditioned on a virtual configuration.
    """

    # Computes the jump operator given the virtual configuration
    @abstractmethod
    def jump(self, virtual_configuration):
        pass

    # Propagates the VState object forward until time t
    @abstractmethod
    def propagate(self, virtual_state: VState, t: float, tlist: list = None):
        pass


class VPropHTD(AVirtualPropagator):
    """
    A propagator that uses qutip.mesolve with time-dependent Hermitian and time-independent non-Hermitian evolution.
    """

    def __init__(self,
                 hamiltonian: list,
                 collapse_operators: list[Qobj] = None,
                 jumps: list[Qobj] = None,
                 expect_operators: Union[Qobj, callable] = None,
                 options: Options = None
                 ):
        """

        :param hamiltonian: a list of the form [Qobj, [Qobj, function], ...] describing the time-dependent HamiltonianBase.
        :param collapse_operators: a list of Qobj describing all the collapse operators.
        :param jumps: a list of Qobj superoperators describing the jump statistics (without scaling by vconfig)
        :param expect_operators: a list of Qobj to evaluate expecation values for
        :param options: an Options object for mesolve.
        """
        self.hamiltonian = hamiltonian

        self.collapse_operators = [] if collapse_operators is None else collapse_operators
        self.collapse_operators = [op[0] if isinstance(op, list) and len(op) == 1 else op for op in
                                   self.collapse_operators]
        self.collapse_operators = [op for op in self.collapse_operators if op != 0 * op]

        self.jumps = [] if jumps is None else jumps
        self.expect_operators = expect_operators
        self.options = options

    def jump(self, vconfig):
        default = 0 * spre(self.hamiltonian[0])
        return sum([-vconfig[i] * list_get(self.jumps, i, default) for i in range(0, len(vconfig))], default)

    def propagate(self, virtual_state: VState, t: float, tlist: list = None):
        jump = self.jump(virtual_state.virtual_configuration)
        if jump != 0 * jump:
            c_ops = self.collapse_operators + [jump]
        else:
            c_ops = self.collapse_operators
        result = mesolve(H=self.hamiltonian,
                         rho0=virtual_state,
                         tlist=[virtual_state.time, t] if tlist is None else tlist,
                         c_ops=c_ops,
                         e_ops=self.expect_operators,
                         options=self.options)
        virtual_state.__init__(state=result.states[-1], time=t,
                               virtual_configuration=virtual_state.virtual_configuration)
        return result


class VPropNHTD(AVirtualPropagator):
    """
    A propagator that uses qutip.mesolve with a time-dependent non-Hermitian evolution.
    """

    def __init__(self,
                 generator: EvaluatedOperator,
                 jumps: list[EvaluatedOperator] = None,
                 expect_operators: Union[Qobj, callable] = None,
                 options: Options = None
                 ):
        """

        :param generator: an EvaluatedOperator object describing the time-dependent generator.
        :param jumps: a list of EvaluatedOperator objects describing possibly time-dependent jumps.
        :param expect_operators: a list of Qobj to evaluate expecation values for
        :param options: an Options object for mesolve.
        """
        self.generator = generator
        self.jumps = [] if jumps is None else jumps
        self.expect_operators = expect_operators
        self.options = options

    def jump(self, vconfig) -> EvaluatedOperator:
        default = 0 * self.generator.constant
        return sum((-vconfig[i] * list_get(self.jumps, i, default) for i in range(0, len(vconfig))), default)

    def propagate(self, virtual_state: VState, t: float, tlist: list = None):
        gen = self.generator + \
              self.jump(virtual_state.virtual_configuration) if virtual_state.virtual_configuration else self.generator
        result = mesolve(H=gen.list_form(),
                         rho0=virtual_state,
                         tlist=[virtual_state.time, t] if tlist is None else tlist,
                         e_ops=self.expect_operators,
                         options=self.options)
        virtual_state.__init__(state=result.states[-1],
                               time=t,
                               virtual_configuration=virtual_state.virtual_configuration)
        return result


class VPropTI(AVirtualPropagator):
    """
    A propagator that uses matrix exponentiation to propagate a state using a time-independent generator
    """

    def __init__(self,
                 generator: Qobj,
                 jumps: list[Qobj] = None,
                 ):
        assert generator.isoper or generator.issuper, "gen must be an operator or superoperator"
        self.generator = liouvillian(generator) if generator.isoper else generator
        self.jumps = [] if jumps is None else jumps

    def jump(self, vconfig):
        default = 0 * self.generator
        return sum([-vconfig[i] * list_get(self.jumps, i, default) for i in range(0, len(vconfig))], default)

    def propagate(self, virtual_state: VState, t: float, tlist: list = None):
        virtual_state.apply_generator(op=(self.generator + self.jump(virtual_state.virtual_configuration)),
                                      time=t - virtual_state.time)
        virtual_state.time = t


def list_get(lst, idx, default):
    try:
        return lst[idx]
    except IndexError:
        return default
