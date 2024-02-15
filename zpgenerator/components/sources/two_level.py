from ...elements import Emitter
from ...time import TimeInterval
from ...dynamic.control import Control
from ...dynamic.pulse import Pulse
from .base_source import GatedSourceComponent
from typing import Union


class TwoLevelSource(GatedSourceComponent):
    """
    A two-level emitter driven by a single pulse
    """

    def __init__(self,
                 pulse: Pulse = None,
                 gate: Union[TimeInterval, list] = None,
                 efficiency: float = 1,
                 parameters: dict = None,
                 name: str = None,
                 emitter_name: str = None):

        emitter = Emitter.two_level(parameters=parameters, name=emitter_name)
        pulse = pulse if pulse else Pulse.dirac(parameters=parameters)

        emitter.add(Control.drive(pulse=pulse, transition=emitter.operators['lower']))

        emitter.initial_state = emitter.states['|g>']

        gate = TimeInterval.source_gate(pulse, parameters=parameters) if gate is None else gate

        super().__init__(emitter=emitter, gate=gate, efficiency=efficiency, name=name, parameters=parameters)
        self.default_name = '_TLS'
