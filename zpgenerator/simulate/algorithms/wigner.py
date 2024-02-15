from .lifetime import Lifetime
from ..base_processor import ProcessorBase
from ...network import AComponent, make_masked_source
from ...virtual import ParityDetectorGate
from ...elements import BeamSplitter, ShapedLaserEmitter
from typing import Union, List
from numpy import sqrt, arcsin, ndarray, pi, real


class WignerFunction:
    """
    A list of complex values and associated Wigner function value
    """

    def __init__(self, alphas: list, points: list):
        self.alphas = alphas
        self.points = [real(p) for p in points]


def compute_wigner_function(source: AComponent, port: int, lifetime: Lifetime, alpha: Union[complex, List[complex]],
                            parameters: dict = None,
                            pseudo_limit: float = 0.01, lo_resolution=600, lo_fluctuations=2):
    """
    :param source: a source of light
    :param port: the port of the source being analysed
    :param lifetime: the shape of the local oscillator (matching the source lifetime)
    :param alpha: a complex amplitude in phase space or a list of such amplitudes
    :param parameters: a dictionary of parameters to modify the default parameters
    :param pseudo_limit: a loss regime parameter for the pseudo-Wigner algorithm
    :param lo_resolution: the number of numerical points to interpolate the local oscillator shape
    :param lo_fluctuations: the maximum nonlinear fluctuation of local oscillator photons
    :return: the value W(alpha) of the Wigner function at the point alpha
    """
    alpha = alpha if isinstance(alpha, list) or isinstance(alpha, ndarray) else [alpha]

    lo_source = ShapedLaserEmitter(truncation=lo_fluctuations, parameters=parameters, shape=lifetime,
                                   resolution=lo_resolution)
    lo_source.initial_state = lo_source.states['|0>']

    p = ProcessorBase()
    p.add(0, make_masked_source(source, port))
    p.add(1, lo_source)
    p.add(0, BeamSplitter(parameters={'angle': arcsin(sqrt(pseudo_limit))}))
    p.add(0, ParityDetectorGate())

    points = [p.probs(parameters={'amplitude': a / sqrt(2 * pseudo_limit / (1 - pseudo_limit))} |
                                 ({} if parameters is None else source.set_parameters(parameters)))['p',] / pi
              for a in alpha]

    return WignerFunction(alphas=alpha, points=points)
