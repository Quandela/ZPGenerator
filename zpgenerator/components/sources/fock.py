from ...elements import ShapedCavityEmitter
from .base_source import GatedSourceComponent
from ...time import TimeInterval, PulseBase, Lifetime
from ...time.parameters import parinit
from typing import Union
from qutip import Qobj, fock


class FockSource(GatedSourceComponent):
    """
    A source of Fock states
    """

    def __init__(self,
                 state: Union[int, Qobj],
                 gate: Union[TimeInterval, list] = None,
                 shape: Union[PulseBase, Lifetime] = None,
                 shape_resolution: int = 1000,
                 efficiency: float = 1,
                 parameters: dict = None,
                 name: str = None):
        truncation = state + 1 if isinstance(state, int) else state.shape[0]

        emitter = ShapedCavityEmitter(shape=shape, resolution=shape_resolution,
                                      truncation=truncation, parameters=parameters)

        emitter.initial_state = fock(truncation, state) if isinstance(state, int) else state

        parameters = parinit({'delay': 0., 'decay': 1.}, parameters)

        if gate is None and shape is None:
            def window(args: dict):
                return [args['delay'], args['delay'] + 15 / args['decay']]

            gate = TimeInterval(interval=window, parameters=parameters)
        else:
            gate = emitter.system.interval

        super().__init__(emitter=emitter, gate=gate, efficiency=efficiency, parameters=parameters, name=name)
        self.default_name = '_|' + (str(state) if isinstance(state, int) else 'psi') + '>'
