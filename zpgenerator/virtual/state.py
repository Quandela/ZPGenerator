from qutip import Qobj, liouvillian
from ..time.evaluate import EvaluatedDiracOperator, expmv
from typing import Union


class VState(Qobj):
    """
    A virtual state of a source conditioned on a history of virtual configurations,
    and that can evolve in time conditioned on a current configuration.

    :param state: a state of the source
    """

    def __init__(self, state: Qobj, time: float = 0, virtual_configuration: list = None):
        super().__init__(inpt=state)
        self.virtual_configuration = [] if virtual_configuration is None else virtual_configuration
        self.time = time

    # Apply an instantaneous operator or superoperator
    def apply_operator(self, op: Union[Qobj, EvaluatedDiracOperator]):
        if isinstance(op, Qobj):
            if op.isoper:
                rho = self if self.isoper else self * self.dag()
                super().__init__(inpt=op * rho * op.dag())
            else:
                super().__init__(inpt=op(self))
            return self
        elif isinstance(op, EvaluatedDiracOperator):
            if op.hamiltonian:
                self.apply_generator(op.hamiltonian)
            if op.channel:
                self.apply_operator(op.channel)

    # Apply an instantaneous HamiltonianBase or Liouvillian
    def apply_generator(self, op: Qobj, time: float = 1):
        # Could still be optimised...
        if op.isoper:
            op = liouvillian(op)

        rho = self if self.isoper else self * self.dag()
        super().__init__(inpt=expmv(time, op, rho), dims=rho.dims)
        return self

    # Propagating the state forward in time given the current configuration
    def propagate(self, propagator, t: float, tlist: list = None):
        return propagator.propagate(self, t, tlist=tlist)
