from ...elements import Emitter
from .base_source import GatedSourceComponent
from ...time import TimeInterval, Operator, PulseBase
from ...time.parameters import parinit
from ...dynamic.control import Control
from ...dynamic.pulse import Pulse
from typing import Union


class BiexcitonSource(GatedSourceComponent):
    """
    A biexciton emitter driven by a single polarized pulse near resonant with the g -> X or X -> XX transition
    """

    def __init__(self,
                 pulse: PulseBase = None,
                 gate: Union[TimeInterval, list, callable] = None,
                 efficiency: float = 1,
                 parameters: dict = None,
                 name: str = None):

        emitter = Emitter.biexciton(parameters=parameters)
        pulse = Pulse.square(parameters=parinit({'detuning': 0}, parameters)) if pulse is None else pulse

        x = emitter.operators['lower_x']
        y = emitter.operators['lower_y']
        bx = emitter.operators['lower_bx']
        by = emitter.operators['lower_by']
        dipole = Operator.polarised((x + bx), (y + by), parinit({'theta': 0, 'phi': 0}, parameters))

        emitter.add(Control.drive(pulse=pulse, transition=dipole))

        emitter.initial_state = emitter.states['|g>']

        gate = TimeInterval.source_gate(pulse, parameters=parameters) if gate is None else gate

        super().__init__(emitter=emitter, gate=gate, efficiency=efficiency, parameters=parameters, name=name)
        self.default_name = '_Biexciton'
