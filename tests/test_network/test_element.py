from zpgenerator.network.element import *
from zpgenerator.system import ScattererBase, HamiltonianBase, EnvironmentBase, EmitterBase
from zpgenerator.time import Operator, TimeOperator, CompositeTimeOperator
from qutip import Qobj, qeye, destroy, create, liouvillian, tensor, fock
from numpy import sin, cos, pi


def test_comp_coll_init_empty():
    comp = ElementCollection()
    assert comp.modes == 0
    assert comp.elements == {}
    assert comp.times() == []
    assert comp.evaluate_quadruple(0).evaluate(0) == Qobj()


def test_comp_coll_init():
    sc = ScattererBase(Operator([[1, 2], [3, 4]]))
    comp = ElementCollection(sc)
    assert comp.modes == 2
    assert comp.elements == {'_ScattererBase': sc}
    assert comp.times() == []
    assert comp.evaluate_quadruple(0).evaluate(0) == Qobj()
    assert comp.evaluate_quadruple(0).scatterer.evaluate(0) == sc.evaluate(0)


def test_comp_coll_add_emitter():
    emitter = EmitterBase(hamiltonian=HamiltonianBase(create(2) * destroy(2)),
                          environment=EnvironmentBase([destroy(2), destroy(2)]),
                          transitions=[destroy(2), destroy(2)],
                          name='QD')

    comp = ElementCollection()
    comp.add(emitter)

    assert comp.modes == 2
    assert comp.elements == {'QD': emitter}
    assert comp.times() == []
    assert comp.parameters == []
    assert comp.parameter_tree() == {}
    assert comp.evaluate_quadruple(0).evaluate(0) == liouvillian(create(2) * destroy(2), c_ops=[destroy(2), destroy(2)])


def test_comp_coll_cascade_scatterer():
    comp = ElementCollection()
    comp.add(ScattererBase(Operator([[1, 2], [3, 4]])))
    comp.add(ScattererBase(Operator([[5, 6], [7, 8]])))

    assert comp.evaluate_quadruple(0).scatterer.evaluate(0) == Qobj([[5, 6], [7, 8]]) * Qobj([[1, 2], [3, 4]])

    comp.add(ScattererBase(Operator([[9, 10], [11, 12]])))

    assert comp.evaluate_quadruple(0).scatterer.evaluate(0) == \
           Qobj([[9, 10], [11, 12]]) * Qobj([[5, 6], [7, 8]]) * Qobj([[1, 2], [3, 4]])


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


def test_comp_coll_cascade_time_dependent_scatterer():
    bs = _make_time_dependent_bs()

    assert bs.evaluate(0) == qeye(2)
    assert bs.evaluate(pi / 2) == 1.j * (destroy(2) + create(2))
    assert bs.evaluate(pi, parameters={'a': 3 / 2}) == -1.j * (destroy(2) + create(2))

    comp = ElementCollection()
    comp.add(ScattererBase(bs))
    comp.add(ScattererBase(bs))
    comp.add(ScattererBase(bs))

    quad = comp.evaluate_quadruple(0)
    assert quad.scatterer.evaluate(pi / 2) == 1.j * (destroy(2) + create(2)) * (-qeye(2))


def test_comp_coll_cascade_emitter_and_scatterer():
    emitter = EmitterBase(hamiltonian=HamiltonianBase(create(2) * destroy(2)),
                          environment=EnvironmentBase([destroy(2), destroy(2)]),
                          transitions=[destroy(2), destroy(2)],
                          name='QD')

    comp = ElementCollection()
    comp.add(ScattererBase(Operator([[1, 2], [3, 4]])))
    comp.add(emitter)
    comp.add(ScattererBase(Operator([[5, 6], [7, 8]])))

    assert [trn.evaluate(0) for trn in comp.evaluate_quadruple(0).transitions] == [(5 + 6) * destroy(2),
                                                                                   (7 + 8) * destroy(2)]
    assert comp.evaluate_quadruple(0).scatterer.evaluate(0) == Qobj([[5, 6], [7, 8]]) * Qobj([[1, 2], [3, 4]])

    comp.add(ScattererBase(Operator([[1, 2], [3, 4]])))

    assert [trn.evaluate(0) for trn in comp.evaluate_quadruple(0).transitions] == [(11 + 2 * 15) * destroy(2),
                                                                                   (3 * 11 + 4 * 15) * destroy(2)]
    assert comp.evaluate_quadruple(0).scatterer.evaluate(0) == \
           Qobj([[1, 2], [3, 4]]) * Qobj([[5, 6], [7, 8]]) * Qobj([[1, 2], [3, 4]])


hamiltonian_test = tensor(create(2) * destroy(2), qeye(2)) + tensor(qeye(2), create(2) * destroy(2))


def test_comp_coll_cascade_emit_scatter_emit():
    emitter0 = EmitterBase(hamiltonian=HamiltonianBase(create(2) * destroy(2)),
                           environment=EnvironmentBase([destroy(2), destroy(2)]),
                           transitions=[destroy(2), create(2)],
                           name='QD 0')
    emitter0.initial_state = fock(2, 0)
    emitter0.initial_time = 1
    assert emitter0.initial_state == fock(2, 0)

    emitter1 = EmitterBase(hamiltonian=HamiltonianBase(create(2) * destroy(2)),
                           environment=EnvironmentBase([destroy(2), create(2)]),
                           transitions=[destroy(2), create(2)],
                           name='QD 1')
    emitter1.initial_state = fock(2, 1)
    emitter1.initial_time = 0

    comp = ElementCollection()
    comp.add(emitter0)
    comp.add(ScattererBase(Operator([[1, 2], [3, 4]])))
    comp.add(emitter1)

    assert comp.initial_state == tensor(fock(2, 0), fock(2, 1))
    assert comp.initial_time == 0
    emitter1.initial_time = 1
    assert comp.initial_time == 1
    assert comp.evaluate_quadruple(0).hamiltonian.evaluate(0) == hamiltonian_test
    assert [env.evaluate(0) for env in comp.evaluate_quadruple(0).environment][:-1] == [tensor(destroy(2), qeye(2)),
                                                                                        tensor(destroy(2), qeye(2)),
                                                                                        tensor(qeye(2), destroy(2)),
                                                                                        tensor(qeye(2), create(2))]
    assert [trn.evaluate(0) for trn in comp.evaluate_quadruple(0).transitions] == \
           [tensor(qeye(2), destroy(2)) + tensor(destroy(2), qeye(2)) + 2 * tensor(create(2), qeye(2)),
            tensor(qeye(2), create(2)) + 3 * tensor(destroy(2), qeye(2)) + 4 * tensor(create(2), qeye(2))]
    assert comp.evaluate_quadruple(0).scatterer.evaluate(0) == Qobj([[1, 2], [3, 4]])
