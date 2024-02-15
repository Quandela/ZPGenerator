from ...elements import Emitter
from .base_source import GatedSourceComponent
from ...time import TimeInterval
from ...dynamic.control import Control
from ...dynamic.pulse import Pulse
from numpy import arccos, sqrt, pi, diag
from qutip import Qobj


class DistinguishableNoiseSource(GatedSourceComponent):
    """
    A source that mimics the distinguishable noise model used in Perceval
    """

    def __init__(self,
                 emission_probability: float = 1,
                 multiphoton_component: float = 0,
                 indistinguishability: float = 1,
                 name: str = None):
        deph = 1 / indistinguishability - 1
        emitter = Emitter.two_level(parameters={'dephasing': 0}, name='tls')

        g2 = multiphoton_component
        beta = emission_probability
        avg = (1 - sqrt(1 - 2 * g2 * beta)) / g2 if g2 != 0 else beta
        theta1 = 2 * arccos((sqrt(2 - avg * (1 + sqrt(1 - 2 * g2)))) / sqrt(2)) if beta != 1 else pi
        theta2 = 2 * arccos((sqrt(2 - avg * (1 - sqrt(1 - 2 * g2)))) / sqrt(2)) if g2 != 0 else 0

        pulse = Pulse(name='excite')
        pulse.add(Pulse.dirac(parameters={'area': theta1, 'delay': 0}, name='p1'))
        pulse.add(Pulse.dirac(parameters={'area': theta2, 'delay': 26}, name='p2'))
        emitter.add(Control.modulate(pulse=pulse, hamiltonian=emitter.operators['X'] / 2))

        pulse_env = Pulse.square(parameters={'width': 26, 'delay': 13, 'area': 26}, name='dephase switch')
        emitter.add(Control.modulate(pulse=pulse_env, environment=sqrt(deph) * emitter.operators['number']))

        pulse_decho = Pulse(name='dephase')
        pulse_decho.add(Pulse.dirac(parameters={'area': 1, 'delay': 0}, name='p1'))
        pulse_decho.add(Pulse.dirac(parameters={'area': 1, 'delay': 26}, name='p2'))
        dephasing_channel = Qobj(diag([1, 0, 0, 1]), dims=[[[2], [2]], [[2], [2]]], type='super')
        emitter.add(Control.operator(pulse=pulse_decho, operator=dephasing_channel))

        emitter.initial_state = emitter.states['|g>']
        emitter.initial_time = 0

        def window(args: dict):
            return [args['delay'], args['delay'] + 52 / args['decay']]

        gate = TimeInterval(interval=window, parameters={'delay': 0., 'decay': 1.}, name='gate')

        parameters = {'tls/dephasing': 0,
                      'gate/delay': 0, 'gate/decay': 1,
                      'p1/delay': 0, 'p2/delay': 26,
                      'excitation/p1/area': theta1, 'excitation/p2/area': theta2,
                      'dephase/p1/area': 1, 'dephase/p2/area': 1,
                      'dephase switch/width': 26, 'dephase switch/delay': 13, 'deph switch/area': 26}

        super().__init__(emitter=emitter, gate=gate, parameters=parameters, name=name)
        self.default_name = '_PCVL'
