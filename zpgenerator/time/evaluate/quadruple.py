from .operator import EvaluatedOperator, evop_mv, evop_umv
from typing import List
from qutip import qzero, qeye, Qobj, liouvillian
from copy import deepcopy


class EvaluatedQuadruple:
    """
    A list of EvaluatedOperators representing an evaluated component
    """

    def  __init__(self,
                 hamiltonian: EvaluatedOperator = None,
                 environment: List[EvaluatedOperator] = None,
                 transitions: List[EvaluatedOperator] = None,
                 scatterer: EvaluatedOperator = None):

        self.hamiltonian = EvaluatedOperator() if hamiltonian is None else hamiltonian
        self.environment = [] if environment is None else environment
        self.transitions = [] if transitions is None else transitions

        if hamiltonian:
            self._subdims = hamiltonian.subdims
        else:
            if self.environment:
                self._subdims = self.environment[0].subdims
            elif self.transitions and isinstance(self.transitions[0], EvaluatedOperator):
                self._subdims = self.transitions[0].subdims
            else:
                self._subdims = [0]

        if not self.transitions:
            if scatterer:
                self.transitions = [EvaluatedOperator(constant=Qobj(inpt=0))] * scatterer.dim

        self.scatterer = EvaluatedOperator(constant=qeye(self.modes)) \
            if scatterer is None else scatterer

        assert self.modes == self.scatterer.dim, \
            "Scattering matrices must have a dimension matching the number of modes"

    @property
    def modes(self):
        return len(self.transitions)

    @property
    def subdims(self):
        return self._subdims

    def tensor_insert(self, i: int, dims: list):
        """
        insert into a larger space
        :param i: position in subdims of new space
        :param dims: list of subdims/dims
        return EvaluatedQuadruple with new dimensions
        """
        return EvaluatedQuadruple(hamiltonian=self.hamiltonian.tensor_insert(i, dims),
                                  environment=[env.tensor_insert(i, dims) for env in self.environment],
                                  transitions=[tra.tensor_insert(i, dims) for tra in self.transitions],
                                  scatterer=self.scatterer)

    def __add__(self, other):
        if other == 0:
            return self
        else:
            return EvaluatedQuadruple(hamiltonian=self.hamiltonian + other.hamiltonian,
                                      environment=self.environment + other.environment,
                                      transitions=self.transitions + other.transitions,
                                      scatterer=self.scatterer.concatenate(other.scatterer))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    # cascaded quantum coupling (https://www.tandfonline.com/doi/full/10.1080/23746149.2017.1343097)
    def __mul__(self, other):
        if other == 1:
            return self
        else:
            assert self.modes == other.modes, "Components must have the same number of modes."
            subdims = [self.subdims, other.subdims]

            scatterer = other.scatterer

            self_vector = [trn.tensor_insert(0, subdims) for trn in self.transitions]
            other_vector = [trn.tensor_insert(1, subdims) for trn in other.transitions]

            hamiltonian = self.hamiltonian.tensor_insert(0, subdims) + other.hamiltonian.tensor_insert(1, subdims)

            environment = [env.tensor_insert(0, subdims) for env in self.environment] + \
                          [env.tensor_insert(1, subdims) for env in other.environment]

            transitions = evop_mv(scatterer, self_vector)
            transitions = [transitions[i] + other_vector[i] for i in range(0, len(transitions))]

            # Quantum cascaded interaction superoperator
            if (not any(v.constant == Qobj([[0]]) for v in other_vector)) and \
                    (not any(v.constant == Qobj([[0]]) for v in self_vector)):
                # for v in other_vector:
                #     print(v.constant)
                #     for pair in v.variable:
                #         print(pair.op)
                LRB = [v.dag().spost() - v.dag().spre() for v in other_vector]
                RLB = [v.spre() - v.spost() for v in other_vector]
                environment += [evop_umv(LRB, scatterer, [v.spre() for v in self_vector]) +
                                evop_umv(RLB, scatterer.dag(), [v.dag().spost() for v in self_vector])]

            # reshape() takes subdims with unit dimensions: [1, 2, 1] and changes it to [2]
            hamiltonian.reshape()
            for env in environment:
                env.reshape()
            for trn in transitions:
                trn.reshape()

            return EvaluatedQuadruple(hamiltonian=hamiltonian, environment=environment,
                                      transitions=transitions, scatterer=other.scatterer * self.scatterer)

    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            assert False, "Cannot cascade backwards"


    def evaluate(self, t: float, parameters: dict = None) -> Qobj:
        h = self.hamiltonian.evaluate(t, parameters)
        c = [env.evaluate(t, parameters) for env in self.environment]
        return liouvillian(H=h if h.isoper else qzero(self.environment[0].subdims),
                           c_ops=[op for op in c if op.isoper]) + sum([op for op in c if op.issuper]) if h.isoper or c \
            else Qobj()

    def pad(self, number: int):
        mode_increase = number - self.modes
        if mode_increase > 0:
            self.scatterer = self.scatterer.concatenate(EvaluatedOperator.id(mode_increase))
            self.transitions += [EvaluatedOperator(qzero(self.subdims)) for i in range(0, mode_increase)]

    def permute(self, perm: List[int]):
        if sorted(perm) != perm:
            self.scatterer = self.scatterer.permute(perm)
            self.transitions = [self.transitions[i] for i in perm]

    def match(self, perm: List[int]):
        quad = deepcopy(self)
        quad.pad(len(perm))
        quad.permute(perm)
        return quad
