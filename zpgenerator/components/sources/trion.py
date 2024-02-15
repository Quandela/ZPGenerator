from ...elements import Emitter
from .base_source import GatedSourceComponent
from ...time import TimeInterval, Operator, PulseBase
from ...time.parameters import parinit
from ...dynamic.control import Control
from ...dynamic.pulse import Pulse
from typing import Union
from numpy import sqrt, pi


class TrionSource(GatedSourceComponent):
    """
    An exciton emitter driven by a single polarized pulse
    """

    def __init__(self,
                 charge: str = 'negative',
                 pulse: PulseBase = None,
                 pulse_orthogonal: PulseBase = None,
                 gate: Union[TimeInterval, list, callable] = None,
                 efficiency: float = 1,
                 parameters: dict = None,
                 name: str = None):
        emitter = Emitter.trion(parameters=parameters, charge=charge)
        pulse = Pulse.dirac(parameters=parameters) if pulse is None else pulse


        right = emitter.operators['lower_R']
        left = emitter.operators['lower_L']
        horizontal = (right + left) / sqrt(2)
        vertical = 1.j * (right - left) / sqrt(2)
        params = parinit({'theta': pi / 4, 'phi': -pi / 2}, parameters)
        dipole = Operator.polarised(horizontal, vertical, parameters=params)
        emitter.add(Control.drive(pulse=pulse, transition=dipole))

        if pulse_orthogonal:
            dipole_orthogonal = Operator.polarised_orthogonal(horizontal, vertical, parameters=params)
            emitter.add(Control.drive(pulse=pulse_orthogonal, transition=dipole_orthogonal))

        emitter.initial_state = (1 / 2) * (emitter.states['|spin_down>'] * emitter.states['|spin_down>'].dag() +
                                           emitter.states['|spin_up>'] * emitter.states['|spin_up>'].dag())

        gate = TimeInterval.source_gate(pulse, parameters=parameters) if gate is None else gate

        super().__init__(emitter=emitter, gate=gate, efficiency=efficiency, name=name)
        self.default_name = '_Trion'
