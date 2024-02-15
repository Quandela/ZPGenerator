from ...system import MultiBodyEmitter, CouplingBase
from .qubit import TwoLevelEmitter
from .cavity import CavityEmitter


class PurcellEmitter(MultiBodyEmitter):
    """
    A two-level system coupled to a cavity mode.
    """

    def __init__(self,
                 purcell_factor: float = None,
                 regime: float = None,
                 timescale: float = None,
                 parameters: dict = None,
                 name: str = None):
        emitter = TwoLevelEmitter(name='emitter')
        cavity = CavityEmitter(name='cavity')
        coupling = CouplingBase.jaynes_cummings(emitter.operators['lower'], cavity.operators['annihilation'])

        super().__init__(subsystems=[emitter, cavity], coupling=coupling, parameters=parameters, name=name)

        def keyword_defaults(args: dict):
            return {'coupling': 1 / (2 * args['regime']) / args['timescale'],
                    'cavity/decay': 1 / args['regime'] ** 2 / args['timescale'],
                    'emitter/decay': 1 / args['purcell_factor'] / args['timescale']}

        make_parameter_function = purcell_factor or regime or timescale

        purcell_factor = purcell_factor if purcell_factor else 10
        regime = regime if regime else 0.1
        timescale = timescale if timescale else 1
        keywords = {'purcell_factor': purcell_factor, 'regime': regime, 'timescale': timescale}

        if make_parameter_function:
            self.create_overwrite_parameter_function(keyword_defaults, parameters=keywords)

        self.update_default_parameters(keyword_defaults(keywords) | (parameters if parameters else {}))
