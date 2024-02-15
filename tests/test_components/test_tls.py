from zpgenerator.elements import Emitter
from zpgenerator.components.sources import SourceComponent
from zpgenerator.components import Source


def test_initial_state_tls_source():
    emitter = Emitter.two_level()
    emitter.initial_state = emitter.states['|g>']
    source = SourceComponent(emitter)
    assert source.initial_state == emitter.states['|g>']
    assert source.subdims == [2]

    source = Source.two_level()
    assert source.subdims == [2]
