from zpgenerator.misc.display import *
from zpgenerator.elements import *
from zpgenerator.components import *


def test_display_elements():
    display(BeamSplitter(name='BS'))
    #    ____
    # --| BS |--
    # --|    |--
    #    ‾‾‾‾

    display(PhaseShifter(name='PS'))
    #    ____
    # --| PS |--
    #    ‾‾‾‾

    display(Emitter.two_level(name='tls'))
    #     _____
    # -- | tls | --
    #     ‾‾‾‾‾
    display(Emitter.exciton(name='exciton'))
    #    _________
    # --| exciton |--
    # --|         |--
    #    ‾‾‾‾‾‾‾‾‾

    display(Emitter.trion(name='trion'))
    #     _______
    # -- | trion | --
    # -- |       | --
    #     ‾‾‾‾‾‾‾

    display(Emitter.biexciton(name='biexciton'))
    #    ___________
    # --|           |--
    # --|           |--
    # --| biexciton |--
    # --|           |--
    #    ‾‾‾‾‾‾‾‾‾‾‾

    display(Emitter.cavity(name='cavity', modes=4))
    #    ________
    # --|        |--
    # --|        |--
    # --| cavity |--
    # --|        |--
    #    ‾‾‾‾‾‾‾‾


def test_display_sources():
    display(Source.two_level(name='two-level emitter'))
    #     ___________________
    # 0--| two-level emitter |--
    #     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

    display(Source.exciton(name='exciton emitter'))
    #     _________________
    # 0--| exciton emitter |--
    # 0--|                 |--
    #     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

    display(Source.trion(name='trion emitter'))
    #     _______________
    # 0--| trion emitter |--
    # 0--|               |--
    #     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

    display(Source.biexciton(name='biexciton emitter'))
    #     ___________________
    # 0--|                   |--
    # 0--|                   |--
    # 0--| biexciton emitter |--
    # 0--|                   |--
    #     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾


def test_display_detectors():
    display(Detector.pnr(2, name='PNRD'))
    #     ______
    #  --| PNRD |--D~
    #     ‾‾‾‾‾‾

    display(Detector.threshold(name='TD'))
    #     ____
    #  --| TD |--D~
    #     ‾‾‾‾

    display(Detector.parity(name='W'))
    #     ___
    #  --| W |--D~
    #     ‾‾‾


def test_display_composite_components():
    c = Source.two_level()
    c.add(0, Circuit.bs())
    c.add(0, Detector.pnr(2))
    display(c)
    #     ___________
    # 0--| component |--D~
    #  --|           |--
    #     ‾‾‾‾‾‾‾‾‾‾‾

    c = Circuit.bs()
    c.add(0, Detector.threshold())
    c.add(1, Circuit.bs())
    c.add(2, Detector.pnr(2))
    display(c)
    #    ___________
    # --|           |--D~
    # --| component |--
    # --|           |--D~
    #    ‾‾‾‾‾‾‾‾‾‾‾
