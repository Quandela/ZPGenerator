from ..base_processor import ProcessorBase
from ...network import DetectorGate, AComponent, make_masked_source
from ...elements import BeamSplitter, PhaseShifter
from numpy import pi


def hong_ou_mandel_processor(source: AComponent, port: int, phi: float = 0., efficiency: float = 1.):
    masked_source = make_masked_source(source, port)

    p = ProcessorBase()
    p.component.mask()
    p.add(0, masked_source)
    p.add(1, masked_source)
    p.add(0, PhaseShifter(parameters={'phase': phi}))
    p.add(0, BeamSplitter())
    p.add(0, DetectorGate(resolution=1, efficiency=efficiency))
    p.add(1, DetectorGate(resolution=1, efficiency=efficiency))

    return p


def estimate_hom_visibility(source: AComponent, port: int,
                            pseudo_limit: float, parameters: dict = None, phi: float = pi/4):
    hom = hong_ou_mandel_processor(source, port, phi, pseudo_limit)
    probs = hom.probs(parameters=parameters)  # minimum 2-photon fringe

    # estimate the norm
    norm = probs[1, 0] + probs[0, 1]
    assert norm != 0, "No light detected, normalization is 0"

    return 1 - 8 * probs[1, 1] / (norm ** 2)


def estimate_hom_visibility_with_coherence(source: AComponent, port: int, pseudo_limit: float, parameters: dict = None):
    hom = lambda phi: hong_ou_mandel_processor(source, port, phi, pseudo_limit)
    probs1 = hom(0).probs(parameters=parameters)  # minimum 2-photon fringe
    probs2 = hom(pi / 2).probs(parameters=parameters)  # maximum 2-photon fringe

    # estimate the norm
    norm = probs1[1, 0] + probs1[0, 1]
    assert norm != 0, "No light detected, normalization is 0"

    # lossy limit formulas
    c1 = 2 * abs(abs(probs1[1, 0] - probs2[1, 0])) / norm
    c2 = 4 * abs(probs1[1, 1] - probs2[1, 1]) / (norm ** 2)
    vhom = 1 - 4 * (probs1[1, 1] + probs2[1, 1]) / (norm ** 2)

    return vhom, c1, c2
