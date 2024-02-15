from ..base_processor import ProcessorBase
from ...network import AComponent, make_masked_source
from ...virtual import PhysicalDetectorGate
from math import factorial
from scipy.special import binom
from collections import UserDict


def compute_average_photon_number(pn: UserDict) -> float:
    return sum(k[0] * v for k, v in pn.items())


def compute_intensity_correlation(pn: UserDict, order: int) -> float:
    mu = compute_average_photon_number(pn)
    return factorial(order) * sum(binom(k[0], order) * v for k, v in pn.items()) / mu ** order


def compute_parity_summation(pn: UserDict) -> float:
    return sum((-1) ** k[0] * v for k, v in pn.items())


def compute_brightness(source: AComponent, port: int, parameters: dict = None):
    p = ProcessorBase(make_masked_source(source, port))
    p.add(0, PhysicalDetectorGate(resolution=1, ignore_zero=True))

    return 1 - p.generating_points(parameters=parameters)[0][0]


def estimate_average_photon_number(source: AComponent, port: int, pseudo_limit: float, parameters: dict = None):
    p = ProcessorBase(make_masked_source(source, port))
    p.add(0, PhysicalDetectorGate(resolution=1, efficiency=pseudo_limit, ignore_zero=True))
    p0 = p.generating_points(parameters=parameters)[0][0]

    # Get the brightness, correct for lossy regime
    return abs((1 - p0) / pseudo_limit)


def estimate_intensity_correlation(source: AComponent, port: int, pseudo_limit: float, parameters: dict = None):
    p = ProcessorBase(make_masked_source(source, port))
    p.add(0, PhysicalDetectorGate(resolution=2, efficiency=pseudo_limit, ignore_zero=True))

    # Get the generating points
    points = p.generating_points(parameters=source.set_parameters(parameters))[0]
    if (1 - abs(points[0])) > 10 ** -8:
        g2 = abs(4 * (1 - 2 * points[1] + points[0]) / (1 - points[0]) ** 2)  # lossy limit formula
        return g2, abs(1 - points[0])/pseudo_limit
    else:
        print("Warning: no light detected in mode " + ('' if source.modes == 1 else str(port)) + ', ' +
              ('g2' if source.modes == 1 else 'g2 ' + str(port)) +
              " cannot be defined.")
        return None, abs(1 - points[0])/pseudo_limit
