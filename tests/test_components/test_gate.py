from zpgenerator.components.switches import Gate
from zpgenerator.system import AQuantumEmitter


def test_gate():
    gate = Gate([0, 2])
    assert gate.times() == [0, 2]
    assert gate.modes == 1
    assert not isinstance(gate, AQuantumEmitter)

