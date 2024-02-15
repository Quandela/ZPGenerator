from ...elements import Emitter
from ...time import TimeInterval
from ...dynamic.control import Control
from ...dynamic.pulse import Pulse
from .base_source import GatedSourceComponent
from typing import Union


class PurcellSource(GatedSourceComponent):
    """
    A two-level emitter driven by a single pulse with emission coupled to a cavity mode
    """

    def __init__(self,
                 pulse: Pulse = None,
                 gate: Union[TimeInterval, list] = None,
                 efficiency: float = 1,
                 purcell_factor: float = None,
                 regime: float = None,
                 timescale: float = None,
                 parameters: dict = None,
                 name: str = None):

        emitter = Emitter.purcell(purcell_factor=purcell_factor, regime=regime, timescale=timescale,
                                  parameters=parameters)
        pulse = pulse if pulse else Pulse.dirac(parameters=parameters)

        emitter.add(Control.drive(pulse=pulse, transition=emitter.operators['lower']))

        emitter.initial_state = emitter.states['|g>|0>']

        if not gate:
            gate = TimeInterval.source_gate(pulse,
                                            parameters=emitter.default_parameters | (parameters if parameters else {}),
                                            parameter_name='_purcell_rate')

            gate.create_insert_parameter_function(self._purcell_rate)

        super().__init__(emitter=emitter, gate=gate, efficiency=efficiency, name=name,
                         parameters=emitter.default_parameters | (parameters if parameters else {}))
        self.default_name = '_Purcell'

    @staticmethod
    def _purcell_rate(args: dict):
        kappa = args['cavity/decay']
        Gamma = args['emitter/decay'] + 2 * args['emitter/dephasing']
        Delta = args['emitter/resonance'] - args['cavity/resonance']
        R = 4 * args['coupling'] ** 2 * (kappa + Gamma) / ((kappa + Gamma) ** 2 + 4 * Delta ** 2)
        return {'_purcell_rate': (R * kappa) / (R + kappa)}

    def canonical_purcell_factor(self, parameters: dict = None):
        parameters = self.default_parameters | (parameters if parameters else {})
        return 4 * parameters['coupling'] ** 2 / (parameters['emitter/decay'] * parameters['cavity/decay'])

    def effective_purcell_factor(self, parameters: dict = None):
        parameters = self.default_parameters | (parameters if parameters else {})
        return self._purcell_rate(parameters)['_purcell_rate'] / parameters['emitter/decay']

    def purcell_inhibition_factor(self, parameters: dict = None):
        return self.effective_purcell_factor(parameters) / self.canonical_purcell_factor(parameters)

    def regime(self, parameters: dict = None):
        parameters = self.default_parameters | (parameters if parameters else {})
        return 2 * parameters['coupling'] / parameters['cavity/decay']

    def timescale(self, parameters: dict = None):
        parameters = self.default_parameters | (parameters if parameters else {})
        return 1 / (parameters['emitter/decay'] * self.canonical_purcell_factor(parameters))

    def effective_lifetime(self, parameters: dict = None):
        parameters = self.default_parameters | (parameters if parameters else {})
        return 1 / (parameters['emitter/decay'] * self.effective_purcell_factor(parameters))
