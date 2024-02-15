from zpgenerator.simulate.quality import *
from zpgenerator.components import Source
from zpgenerator.network import Component
from zpgenerator.dynamic import Pulse
from math import isclose
from numpy import pi, sin, sqrt


def assert_quality(source, expected: dict, port='0'):
    for k, v in expected.items():
        assert isclose(source.quality[port][k], v, abs_tol=1e-3)


def test_quality_source_two_level_dirac():
    p = ProcessorQuality()
    p.add(0, Source.two_level())

    assert p.component.subdims == [2]

    assert p.parameters == ['area', 'decay', 'delay', 'dephasing', 'efficiency', 'phase', 'resonance']
    assert p.default_parameters == {'area': pi,
                                    'decay': 1.0,
                                    'delay': 0,
                                    'dephasing': 0.0,
                                    'resonance': 0.0,
                                    'phase': 0,
                                    'efficiency': 1}

    p.mu()
    assert_quality(p, {'mu': 1})
    p.mu(parameters={'efficiency': 0.5})
    assert_quality(p, {'mu': 0.5})
    p.mu(parameters={'area': pi / 4})
    assert_quality(p, {'mu': sin(pi / 8) ** 2})

    p.beta()
    assert_quality(p, {'beta': 1})
    p.beta(parameters={'efficiency': 0.5})
    assert_quality(p, {'beta': 0.5})
    p.beta(parameters={'area': pi / 4})
    assert_quality(p, {'beta': sin(pi / 8) ** 2})

    p.g2()
    assert_quality(p, {'g2': 0})

    p.hom()
    assert_quality(p, {'vhom': 1, 'M': 1, 'c1': 0, 'c2': 0})

    assert p.component.subdims == [2]

    prbs = p.photon_statistics()
    assert all(isclose(v, 1, abs_tol=1e-5) if k[0] == 1 else isclose(v, 0, abs_tol=1e-5) for k, v in prbs.items())


def test_quality_source_two_level_square():
    p = ProcessorQuality()
    pulse = Pulse.square(parameters={'area': pi, 'width': 1})
    source = Source.two_level(pulse=pulse)
    p.add(0, source)

    assert p.parameters == ['area', 'decay', 'delay', 'dephasing', 'efficiency', 'phase', 'resonance', 'width']
    assert p.default_parameters == {'area': pi,
                                    'decay': 1.0,
                                    'delay': 0,
                                    'dephasing': 0.0,
                                    'resonance': 0.0,
                                    'phase': 0,
                                    'efficiency': 1,
                                    'width': 1}

    p.mu(pseudo_limit=0.01)
    assert_quality(p, {'mu': 1.0730825704101776})
    p.mu(parameters={'efficiency': 0.5})
    assert_quality(p, {'mu': 0.5367798158443349})

    p.beta()
    assert_quality(p, {'beta': 0.9800857377719977})
    p.beta(parameters={'efficiency': 0.5})
    assert_quality(p, {'beta': 0.5133456584623748})

    p.g2(pseudo_limit=0.01)
    assert_quality(p, {'g2': 0.1657173108692473})

    p.hom()
    assert_quality(p, {'vhom': 0.6986076195322675,
                       'M': 0.8632249191373985,
                       'c1': 0.16937487575139845,
                       'c2': 0.010913449702802001})

    prbs = p.photon_statistics(truncation=5)
    prbs_expected = {0: 0.01991494387973506,
                     1: 0.8876103713280139,
                     2: 0.09100574814203072,
                     3: 0.001461263744472416,
                     4: 7.0697479682875946e-06,
                     5: 6.031577793541691e-07}
    assert all(isclose(v, prbs[k], abs_tol=1e-5) for k, v in prbs_expected.items())


def test_quality_source_two_level_gaussian():
    p = ProcessorQuality()
    pulse = Pulse.gaussian(parameters={'area': pi, 'width': 1})
    source = Source.two_level(pulse=pulse)
    p.add(0, source)

    assert p.parameters == ['area', 'decay', 'delay', 'dephasing', 'detuning', 'efficiency', 'phase', 'resonance',
                            'width', 'window']
    assert p.default_parameters == {'area': 3.141592653589793,
                                    'decay': 1.0,
                                    'delay': 0,
                                    'dephasing': 0.0,
                                    'detuning': 0,
                                    'efficiency': 1,
                                    'phase': 0,
                                    'resonance': 0.0,
                                    'width': 1,
                                    'window': 6}

    p.mu(pseudo_limit=0.01)
    assert_quality(p, {'mu': 1.013882376243147})
    p.mu(parameters={'efficiency': 0.5})
    assert_quality(p, {'mu': 0.5073233662694854})

    p.beta()
    assert_quality(p, {'beta': 0.8685226435566074})
    p.beta(parameters={'efficiency': 0.5})
    assert_quality(p, {'beta': 0.47023019146262746})

    p.g2(pseudo_limit=0.01)
    assert_quality(p, {'g2': 0.2974272053102568})

    p.hom()
    assert_quality(p, {'vhom': 0.5537260184028457,
                       'M': 0.8512403085987019,
                       'c1': 0.49738289802967767,
                       'c2': 0.09811570778749941})

    prbs = p.photon_statistics(truncation=5)
    prbs_expected = {0: 0.13147674563955142,
                     1: 0.7276096061477144,
                     2: 0.135030811250973,
                     3: 0.00578726096993834,
                     4: 9.444644109876665e-05,
                     5: 1.1295507235202043e-06}
    assert all(isclose(v, prbs[k], abs_tol=1e-5) for k, v in prbs_expected.items())


def test_quality_source_distinguishable_noise():
    source = Source.perceval(emission_probability=0.5)
    p = ProcessorQuality()
    p.add(0, source)

    p.mu()
    p.beta()
    p.g2()
    assert_quality(p, {'mu': 0.5, 'beta': 0.5, 'g2': 0})

    p.hom()
    assert_quality(p, {'vhom': 1, 'M': 1, 'c1': 0, 'c2': 0})

    source = Source.perceval(multiphoton_component=0.1)
    p = ProcessorQuality()
    p.add(0, source)

    p.mu()
    p.beta()
    p.g2()
    assert_quality(p, {'mu': (1 - sqrt(1 - 2 * 0.1)) / 0.1, 'beta': 1, 'g2': 0.1})

    p.hom(pseudo_limit=0.002)
    assert_quality(p, {'vhom': 1 - 2 * 0.1, 'M': 1 - 0.1, 'c1': 0, 'c2': 0})

    source = Source.perceval(indistinguishability=0.9)
    p = ProcessorQuality()
    p.add(0, source)

    p.mu()
    p.beta()
    p.g2()
    assert_quality(p, {'mu': 1, 'beta': 1, 'g2': 0})

    p.hom(pseudo_limit=0.002)
    assert_quality(p, {'vhom': 0.9, 'M': 0.9, 'c1': 0, 'c2': 0})


def test_multimode_source():
    source = Source.exciton()
    assert source.output.ports[0].is_open
    assert source.output.ports[1].is_open

    c = Component() // source
    assert c.output.ports[0].is_open
    assert c.output.ports[1].is_open

    c.output.ports[1].close()
    assert source.output.ports[1].is_open

    p = ProcessorQuality() // source
    pn0 = p.photon_statistics(0)
    assert isclose(pn0[1], 1, abs_tol=1e-5)
    pn1 = p.photon_statistics(1)
    assert isclose(pn1[1], 0, abs_tol=1e-5)


def test_detuned_pulse():
    pulse = Pulse.gaussian()
    source = ProcessorQuality() // Source.two_level(pulse=pulse)

    pn0 = source.photon_statistics(parameters={'detuning': 1.512})
    pn1 = source.photon_statistics(parameters={'resonance': 1.512})
    assert all(isclose(pn0[i], pn1[i], abs_tol=1e-5) for i in range(3))

    pn0 = source.photon_statistics(parameters={'detuning': -0.42, 'resonance': -0.42})
    pn1 = source.photon_statistics()
    assert all(isclose(pn0[i], pn1[i], abs_tol=1e-5) for i in range(3))
