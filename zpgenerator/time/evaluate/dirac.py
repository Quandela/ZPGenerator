from .tensor import tensor_insert
from functools import cache
from qutip import Qobj, sprepost, qeye


class EvaluatedDiracOperator:
    """
    A Qobj operator HamiltonianBase or superoperator channel (or both) to be applied directly to a state
    """

    def __init__(self, hamiltonian: Qobj = 0, channel: Qobj = 1):
        self.hamiltonian = hamiltonian
        self.channel = channel

    def __add__(self, other):
        return EvaluatedDiracOperator(hamiltonian=self.hamiltonian + other.hamiltonian,
                                      channel=self.channel * other.channel)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return EvaluatedDiracOperator(hamiltonian=self.hamiltonian + other.hamiltonian,
                                          channel=other.channel * self.channel)

    def __mul__(self, other):
        if other == 1:
            return self
        else:
            return self.tensor_insert(0, [self.subdims, other.subdims]) + \
                other.tensor_insert(1, [self.subdims, other.subdims])

    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            return self.__mul__(other)

    @property
    def subdims(self):
        return self.hamiltonian.dims[0]

    def dag(self):
        return EvaluatedDiracOperator(hamiltonian=0 if self.hamiltonian == 0 else self.hamiltonian.dag(),
                                      channel=1 if self.channel == 1 else self.channel.dag())

    def tensor_insert(self, i: int, dims: list):
        return EvaluatedDiracOperator(hamiltonian=0 if self.hamiltonian == 0 else tensor_insert(self.hamiltonian, i, dims),
                                      channel=1 if self.channel == 1 else tensor_insert(self.channel, i, dims))

    # @cache
    def evaluate(self, commute=False) -> Qobj:
        """
        Evaluates the operator to a single channel
        :param commute: default order is to apply HamiltonianBase first, set commute = True to flip this order
        :return: a Qobj superoperator
        """
        ch = self.channel
        op = self.hamiltonian
        if ch == 0 and op == 0:
            return Qobj(0)
        else:
            if ch == 0:
                i = qeye(self.hamiltonian.dims[0][0])
                ch = sprepost(i, i)
            if op == 0:
                i = qeye(self.channel.dims[0][0])
                op = sprepost(i, i)
            else:
                op = unitary_propagation_superoperator(self.hamiltonian)
        return op * ch if commute else ch * op


def unitary_propagation_superoperator(hamiltonian: Qobj) -> Qobj:
    return sprepost((-1.j * hamiltonian).expm(), (1.j * hamiltonian).expm())
