from zpgenerator.components import Source
from zpgenerator.elements import TwoLevelEmitter, CavityEmitter, Emitter
from zpgenerator.system import CouplingBase, MultiBodyEmitter


def test_source_init():
    emitter = TwoLevelEmitter()
    cavity = CavityEmitter()
    coupling = CouplingBase.jaynes_cummings(emitter.operators['lower'], cavity.operators['annihilation'])
    assert emitter.subdims == [2]
    assert cavity.subdims == [2]
    assert coupling.subdims == [2, 2]

    coupling2 = CouplingBase()
    coupling2.add(coupling)
    assert coupling2.subdims == [2, 2]

    purcell = MultiBodyEmitter(subsystems=[emitter, cavity], coupling=coupling)
    assert purcell.coupling.subdims == [2, 2]

    emitter = Emitter.purcell()
    assert emitter.subdims == [2, 2]

    source = Source.purcell()
    assert source.subdims == [2, 2]




