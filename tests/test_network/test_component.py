from zpgenerator.network.component import *
from zpgenerator.network.detector import DetectorGate
from zpgenerator.elements.linear import BeamSplitter
from zpgenerator.elements import Emitter
from zpgenerator.time import Operator, TimeOperator, CompositeTimeOperator
from zpgenerator.system import ScattererBase, HamiltonianBase, EnvironmentBase, EmitterBase
from zpgenerator.virtual.configuration import PhysicalDetectorGate
from qutip import Qobj, qzero, qeye, destroy, create, tensor, fock
from numpy import pi, sqrt, cos, sin
from copy import deepcopy
from pytest import raises
from zpgenerator.time.parameters import Parameters


d = Parameters.DELIMITER

def test_component_scattering():
    comp = Component()
    comp.add(ScattererBase(Operator([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))

    comp.output.ports[1].close()
    assert comp.input_modes == 3
    assert comp.output_modes == 2

    s = ScattererBase(Operator([[10, 11], [12, 13]]))
    comp.add(s)
    assert s.input_modes == 2

    assert [port.is_closed for port in comp.input.ports] == [False, False, False]
    assert [port.is_closed for port in comp.output.ports] == [False, True, False]
    assert comp.permutations == [[0, 1, 2], [0, 2, 1]]

    quad = comp.evaluate_quadruple(0)
    assert quad.hamiltonian.evaluate(0) == Qobj()
    assert [env.evaluate(0) for env in quad.environment] == []
    assert [trn.evaluate(0) for trn in quad.transitions] == [qzero(1), qzero(1), qzero(1)]
    assert quad.scatterer.evaluate(0) == \
           Qobj([[10, 0, 11], [0, 1, 0], [12, 0, 13]]) * Qobj([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    comp.add(ScattererBase(Operator([[0, 1], [1, 0]])))
    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) * \
           Qobj([[10, 0, 11], [0, 1, 0], [12, 0, 13]]) * \
           Qobj([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    comp.add(ScattererBase(Operator([[1, 1], [1, 1]])))
    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[1, 0, 1], [0, 1, 0], [1, 0, 1]]) * \
           Qobj([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) * \
           Qobj([[10, 0, 11], [0, 1, 0], [12, 0, 13]]) * \
           Qobj([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_component_add_masked_scattering():
    op = Component()
    op.add(ScattererBase(Operator([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    op.input.ports[0].close()
    op.output.ports[1].close()
    op.output.ports[2].close()

    comp = Component()
    comp.add(op)
    assert comp.input.open_modes == 2
    assert comp.output.open_modes == 1
    assert comp.input.closed_modes == 1
    assert comp.output.closed_modes == 2

    # with raises(Exception):
    #     comp.add(op)

    comp.add(ScattererBase(Operator([[2]])))
    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[2, 0, 0], [0, 1, 0], [0, 0, 1]]) * Qobj([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    op = Component()
    op.add(ScattererBase(Operator([[10, 11], [12, 13]])))
    op.input.ports[0].close()

    comp.add(op)

    assert comp.permutations == [[0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 0]]
    assert [port.is_closed for port in comp.input.ports] == [True, False, False, True]
    assert [port.is_closed for port in comp.output.ports] == [False, True, True, False]

    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[13, 0, 0, 12], [0, 1, 0, 0], [0, 0, 1, 0], [11, 0, 0, 10]]) * \
           Qobj([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) * \
           Qobj([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]])


def test_component_emitters():
    emitter = EmitterBase(hamiltonian=HamiltonianBase(create(2) * destroy(2)),
                          environment=EnvironmentBase([destroy(2), create(2), destroy(2)]),
                          transitions=[destroy(2), create(2), create(2)],
                          name='QD 0')
    emitter.initial_time = 1
    emitter.initial_state = fock(2, 1)

    source = Component()
    source.add(emitter)
    source.input.ports[0].close()
    source.input.ports[1].close()
    source.input.ports[2].close()
    source.output.ports[1].close()

    assert source.modes == 3
    assert source.input_modes == 0
    assert source.output_modes == 2

    source.add(ScattererBase(Operator([[1, 2], [3, 4]])))
    assert source.permutations == [[0, 1, 2], [0, 2, 1]]

    assert source.modes == 3
    assert source.input_modes == 0
    assert source.output_modes == 2

    masked_emitter = Component()
    masked_emitter.add(emitter)
    masked_emitter.input.ports[2].close()
    masked_emitter.output.ports[0].close()
    masked_emitter.output.ports[2].close()

    source.add(masked_emitter)
    assert source.permutations == [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]
    assert source.initial_state == tensor(fock(2, 1), fock(2, 1))
    assert source.initial_time == 1
    assert source.modes == 4
    assert source.input_modes == 0
    assert source.output_modes == 1

    quad = source.evaluate_quadruple(0)
    assert quad.hamiltonian.evaluate(0) == tensor(create(2) * destroy(2), qeye(2)) + \
           tensor(qeye(2), create(2) * destroy(2))
    assert [env.evaluate(0) for env in quad.environment][:-1] == \
           [tensor(destroy(2), qeye(2)),
            tensor(create(2), qeye(2)),
            tensor(destroy(2), qeye(2)),
            tensor(qeye(2), destroy(2)),
            tensor(qeye(2), create(2)),
            tensor(qeye(2), destroy(2)),
            ]
    assert [trn.evaluate(0) for trn in quad.transitions] == \
           [tensor(qeye(2), destroy(2)) + tensor(destroy(2), qeye(2)) + 2 * tensor(create(2), qeye(2)),
            tensor(create(2), qeye(2)),
            tensor(qeye(2), create(2)) + 3 * tensor(destroy(2), qeye(2)) + 4 * tensor(create(2), qeye(2)),
            tensor(qeye(2), create(2))]
    assert quad.scatterer.evaluate(0) == Qobj([[1, 0, 2, 0], [0, 1, 0, 0], [3, 0, 4, 0], [0, 0, 0, 1]])


def test_component_unmatched_scattering():
    comp0 = Component()
    comp0.add(ScattererBase(Operator([[1, 2], [3, 4]])))
    comp0.output.ports[1].close()

    comp1 = Component()
    comp1.add(ScattererBase(Operator([[5, 6], [7, 8]])))

    comp0.add(comp1)
    quad = comp0.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[5, 0, 6], [0, 1, 0], [7, 0, 8]]) * Qobj([[1, 2, 0], [3, 4, 0], [0, 0, 1]])
    assert [port.is_closed for port in comp0.output.ports] == [False, True, False]

    comp0 = Component()
    comp0.add(ScattererBase(Operator([[1, 2], [3, 4]])))

    comp1 = Component()
    comp1.add(ScattererBase(Operator([[5, 6], [7, 8]])))
    comp1.input.ports[1].close()

    comp0.add(comp1)
    quad = comp0.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[5, 0, 6], [0, 1, 0], [7, 0, 8]]) * Qobj([[1, 2, 0], [3, 4, 0], [0, 0, 1]])
    assert [port.is_closed for port in comp0.output.ports] == [False, False, False]

    comp0.add(ScattererBase(Operator([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])))
    comp0.input.ports[2].close()

    quad = comp0.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]) * \
           Qobj([[5, 0, 6, 0], [0, 1, 0, 0], [7, 0, 8, 0], [0, 0, 0, 1]]) * \
           Qobj([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert [port.is_closed for port in comp0.output.ports] == [False, False, False, False]


def test_component_add_unmatched_padding():
    comp0 = Component()
    comp0.add(ScattererBase(Operator([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))

    comp1 = Component()
    comp1.add(ScattererBase(Operator([[10, 11], [12, 13]])))

    comp = deepcopy(comp0)
    comp.add(comp1)

    assert comp.permutations == [[0, 1, 2], [0, 1, 2]]
    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[10, 11, 0], [12, 13, 0], [0, 0, 1]]) * Qobj([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    comp = deepcopy(comp1)
    comp.add(comp0)

    assert comp.permutations == [[0, 1, 2], [0, 1, 2]]
    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * Qobj([[10, 11, 0], [12, 13, 0], [0, 0, 1]])


def test_component_init_position():
    comp = Component()
    comp.add(1, ScattererBase(Operator([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    assert comp.modes == 4
    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == Qobj([[1, 0, 0, 0], [0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]])


def test_component_scattering_position():
    comp = Component()
    comp.add(ScattererBase(Operator([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    comp.add(1, ScattererBase(Operator([[10, 11], [12, 13]])))
    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[1, 0, 0], [0, 10, 11], [0, 12, 13]]) * Qobj([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert comp.modes == 3

    comp0 = Component()
    comp0.add(ScattererBase(Operator([[14, 15], [16, 17]])))
    comp0.input.ports[0].close()
    comp.add(0, comp0)

    assert comp.permutations == [[0, 1, 2, 3], [2, 0, 1, 3], [1, 2, 3, 0]]

    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[17, 0, 0, 16], [0, 1, 0, 0], [0, 0, 1, 0], [15, 0, 0, 14]]) * \
           Qobj([[1, 0, 0, 0], [0, 10, 11, 0], [0, 12, 13, 0], [0, 0, 0, 1]]) * \
           Qobj([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]])


def test_component_scattering_position_padding():
    comp = Component()
    comp.add(ScattererBase(Operator([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    comp.add(1, ScattererBase(Operator([[10, 11, 12], [13, 14, 15], [16, 17, 18]])))

    assert comp.permutations == [[0, 1, 2, 3], [3, 0, 1, 2]]
    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(0) == \
           Qobj([[1, 0, 0, 0], [0, 10, 11, 12], [0, 13, 14, 15], [0, 16, 17, 18]]) * \
           Qobj([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]])


def test_component_scattering_position_detector():
    comp = Component()
    comp.add(ScattererBase(Operator([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    comp.add(0, DetectorGate(gate=[0, 1]))
    assert comp._position_to_add(1) == 0
    comp.add(1, ScattererBase(Operator([[10, 11, 12], [13, 14, 15], [16, 17, 18]])))

    assert comp.modes == 4
    assert comp.permutations == [[0, 1, 2, 3], [3, 0, 1, 2]]

    comp.add(0, DetectorGate(gate=[1, 2]))
    assert comp.output.bins == 2

    with raises(Exception):
        comp.add(0, BeamSplitter())
    with raises(Exception):
        comp.add(0, DetectorGate(gate=[0, 1]))

    comp = Component()
    comp.add(0, BeamSplitter())
    comp.add(1, DetectorGate())
    comp.add(0, BeamSplitter())
    assert comp.modes == 3


def _make_time_dependent_bs():
    bs = CompositeTimeOperator(name='BS')

    op = TimeOperator(operator=Qobj([[1, 0], [0, 0]]))
    op.add(lambda t, args: cos(args['a'] * t), parameters={'a': 1})
    bs.add(op)

    op = TimeOperator(operator=Qobj([[0, 1], [0, 0]]))
    op.add(lambda t, args: 1.j * sin(args['a'] * t), parameters={'a': 1})
    bs.add(op)

    op = TimeOperator(operator=Qobj([[0, 0], [1, 0]]))
    op.add(lambda t, args: 1.j * sin(args['a'] * t), parameters={'a': 1})
    bs.add(op)

    op = TimeOperator(operator=Qobj([[0, 0], [0, 1]]), parameters={'a': 1})
    op.add(lambda t, args: cos(args['a'] * t))
    bs.add(op)

    return bs


def test_component_scattering_time():
    bs = _make_time_dependent_bs()
    bs.name = None

    comp0 = Component()
    comp0.add(ScattererBase(bs, name='BS0'))
    comp0.output.ports[1].close()

    comp1 = Component()
    comp1.add(ScattererBase(bs, name='BS1'))

    comp0.add(comp1)

    par = {'BS0' + d + 'a': 1, 'BS1' + d + 'a': 2, 'BS2' + d + 'a': 1 / 2}
    time = pi / 2
    quad = comp0.evaluate_quadruple(time, par)
    assert quad.scatterer.evaluate(time, par) == \
           Qobj([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) * Qobj([[0, 1.j, 0], [1.j, 0, 0], [0, 0, 1]])

    comp2 = Component()
    comp2.add(ScattererBase(bs, name='BS2'))
    comp0.add(comp2)

    assert comp0.parameters == ['BS0' + d + 'a', 'BS1' + d + 'a', 'BS2' + d + 'a']

    par = {'BS0' + d + 'a': 1, 'BS1' + d + 'a': 2, 'BS2' + d + 'a': 1 / 2}
    time = pi / 2
    quad = comp0.evaluate_quadruple(time, par)
    assert quad.scatterer.evaluate(time, par) == \
           Qobj([[1, 0, 1.j], [0, sqrt(2), 0], [1.j, 0, 1]]) / sqrt(2) * \
           Qobj([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) * \
           Qobj([[0, 1.j, 0], [1.j, 0, 0], [0, 0, 1]])


def test_unmasked_component_add():
    p = Component(name='Processor', masked=False)
    p.add(0, BeamSplitter(name='BS 0'))
    p.add(1, BeamSplitter())
    p.add(0, DetectorGate(gate=[0, 1]), bin_name='D0')
    p.add(0, DetectorGate(gate=[1, 2]), bin_name='D1')
    with raises(Exception):
        p.add(0, BeamSplitter())

    p.add(1, BeamSplitter())
    assert p.modes == 3

    p.add(2, BeamSplitter())
    assert p.modes == 4

    p.add(2, DetectorGate())
    assert p.modes == 4

    p.add(1, BeamSplitter())
    assert p.modes == 4


def test_unmasked_component_init_pos():
    p = Component(masked=False)
    p.add(4, BeamSplitter())
    assert p.modes == 6
    p.add(2, DetectorGate())
    p.output.ports[0].close()
    p.output.ports[3].close()
    p.output.ports[4].close()
    p.output.ports[5].close()
    p.add(1, BeamSplitter())
    assert p.modes == 7


def test_unmasked_component_add_component():
    ppnr = Component(name='PPNR', masked=False)
    ppnr.add(1, BeamSplitter())
    ppnr.add(0, BeamSplitter())
    ppnr.add(2, BeamSplitter())
    for i in range(4):
        ppnr.add(i, DetectorGate(), bin_name=str(i))
    ppnr.input.ports[0].close()
    ppnr.input.ports[2].close()
    ppnr.input.ports[3].close()

    assert ppnr.input_modes == 1

    source = Component(Emitter.two_level(modes=2, name='Source'), masked=False)
    source.input.ports[0].close()
    source.input.ports[1].close()

    p = Component(masked=False)
    p.add(0, source)

    assert p.output_modes == 2

    p.add(0, ppnr)
    assert p.modes == 5
    assert p.output_modes == 1
    assert p.input_modes == 0

    p.add(1, ppnr)


def test_initial_state_emitter():
    emitter = Emitter.two_level()
    emitter.initial_state = emitter.states['|g>']
    source = Component(emitter)
    source.add(BeamSplitter())
    assert source.initial_state == emitter.states['|g>']


def test_component_times():
    source = Component(Emitter.two_level())
    source.add(0, PhysicalDetectorGate(resolution=1, efficiency=0.1, ignore_zero=True))
    assert source.times() == []
