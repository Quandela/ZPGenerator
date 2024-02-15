from ..time import TimeOperator, Operator, PulseBase
from ..system import ControlBase, HamiltonianBase, EnvironmentBase
from typing import Union
from qutip import Qobj
from copy import copy


class Control(ControlBase):
    """
    A control object factory
    """

    @classmethod
    def modulate(cls,
                 pulse: PulseBase,
                 hamiltonian: Union[Qobj, Operator] = None,
                 environment: Union[Qobj, Operator] = None,
                 parameters: dict = None,
                 name: str = None):
        """
        A potential modulated by a pulse and/or a corresponding modulated response operator or superoperator.

        :param pulse: a pulse object describing how the potential evolves in time.
        :param hamiltonian: a potential (operator) that is modulated by the pulse (ex. a sigma_x operator).
        :param environment: a Lindblad operator or superoperator that is incidentally modulated by the pulse.
        :param parameters: a dictionary of default parameters applied to dependent parameterised objects.
        :param name: a name to distinguish the parameters.
        :return: a Control object.
        """
        if environment is not None:
            environment = EnvironmentBase(operators=TimeOperator(operator=environment, functions=pulse))
        if hamiltonian is not None:
            hamiltonian = HamiltonianBase(operators=TimeOperator(operator=hamiltonian, functions=pulse))
        return ControlBase(hamiltonian=hamiltonian, environment=environment, parameters=parameters, name=name)

    @classmethod
    def drive(cls,
              pulse: PulseBase,
              transition: Union[Qobj, Operator] = None,
              parameters: dict = None,
              name: str = None):
        """
        A potential modulated by a pulse and/or a corresponding modulated response operator or superoperator.

        :param pulse: a pulse object describing how the potential evolves in time.
        :param transition: a transition operator of the dipole driven by the pulse.
        :param parameters: a dictionary of default parameters applied to dependent parameterised objects.
        :param name: a name to distinguish the parameters.
        :return: a Control object.
        """
        if isinstance(transition, Operator):
            transition = copy(transition)
            if transition.is_callback:
                transition_matrix = transition.matrix
                transition.matrix = lambda args: transition_matrix(args) / 2
            else:
                transition.matrix = transition.matrix / 2
        transition = transition / 2 if isinstance(transition, Qobj) else transition
        hamiltonian = HamiltonianBase(operators=[TimeOperator(operator=transition, functions=pulse),
                                                 TimeOperator(operator=transition, functions=pulse, dag=True)])
        return ControlBase(hamiltonian=hamiltonian, parameters=parameters, name=name)

    @classmethod
    def operator(cls,
                 pulse: PulseBase,
                 operator: Union[Qobj, Operator]):
        """
        A potential modulated by a pulse and/or a corresponding modulated response operator or superoperator.

        :param pulse: a dirac pulse object describing when the channel is applied.
        :param operator: a channel (operator or superoperator) that is applied directly to the state.
        :return: a Control object.
        """
        assert not pulse.has_interval, "Pulse must include only Dirac delta functions."
        assert pulse.has_instant, "Pulse must include at least one Dirac delta."
        return ControlBase(channel=TimeOperator(operator=operator, functions=pulse))
