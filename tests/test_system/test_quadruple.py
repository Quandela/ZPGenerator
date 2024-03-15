from zpgenerator.time.evaluate.quadruple import *
from zpgenerator.time import OpFuncPair, TimeOperator, CompositeTimeOperator
from numpy import sin, cos, pi
from qutip import create, destroy, qeye, tensor, num, spre, spost, lindblad_dissipator
from zpgenerator.time.parameters import Parameters

d = Parameters.DELIMITER


def test_evaluatequad_init_empty():
    quad = EvaluatedQuadruple()
    assert quad.modes == 0
    assert quad.subdims == [0]
    assert quad.hamiltonian.evaluate(0) == Qobj()
    assert quad.environment == []
    assert quad.transitions == []
    assert quad.scatterer.evaluate(0) == qeye(0)
    assert quad.evaluate(0) == Qobj()


def test_evaluatequad_init_qobj():
    ham = EvaluatedOperator(constant=destroy(2))
    quad = EvaluatedQuadruple(hamiltonian=ham)
    assert quad.modes == 0
    assert quad.subdims == [2]
    assert quad.hamiltonian.evaluate(0) == destroy(2)
    assert quad.environment == []
    assert quad.transitions == []
    assert quad.scatterer.evaluate(0) == qeye(0)
    assert quad.evaluate(0) == liouvillian(destroy(2))


def test_evaluatequad_init_env():
    quad = EvaluatedQuadruple(environment=[EvaluatedOperator(constant=destroy(2))])
    assert quad.environment[0].evaluate(0) == destroy(2)
    assert quad.evaluate(0) == liouvillian(qzero(2), [destroy(2)])

    quad.hamiltonian = EvaluatedOperator(constant=create(2) * destroy(2))
    assert quad.hamiltonian.evaluate(0) == create(2) * destroy(2)
    assert quad.evaluate(0) == liouvillian(create(2) * destroy(2), [destroy(2)])


def make_full_quad(n, m):
    ham = EvaluatedOperator(constant=create(n) * destroy(n))
    trn = EvaluatedOperator(constant=destroy(n))
    sct = EvaluatedOperator(constant=qeye(m))
    return EvaluatedQuadruple(hamiltonian=ham, environment=[ham] + [trn] * m, transitions=[trn] * m, scatterer=sct)


def make_full_quad_variable(n, m):
    ham = EvaluatedOperator(constant=create(n) * destroy(n),
                            variable=[OpFuncPair(op=create(n) * destroy(n),
                                                 func=lambda t, args: args['a'] * t ** 2)])
    trn = EvaluatedOperator(constant=destroy(n),
                            variable=[OpFuncPair(op=destroy(n),
                                                 func=lambda t, args: args['b'] * t)])
    sct = EvaluatedOperator(constant=qeye(m))
    return EvaluatedQuadruple(hamiltonian=ham, environment=[ham] + [trn] * m, transitions=[trn] * m, scatterer=sct)


def test_evaluatequad_init_mode():
    quad = make_full_quad(2, 1)
    assert quad.modes == 1
    assert quad.subdims == [2]

    quad = make_full_quad_variable(2, 1)
    assert quad.modes == 1
    assert quad.subdims == [2]


def test_evaluatequad_tensor_constant():
    def _test_modes(m):
        quad = make_full_quad(3, m)
        quad = quad.tensor_insert(1, [2, 3])
        assert quad.modes == m
        assert quad.subdims == [2, 3]
        assert quad.hamiltonian.evaluate(0) == tensor(qeye(2), create(3) * destroy(3))
        assert all(env.subdims == [2, 3] for env in quad.environment)
        assert all(trn.subdims == [2, 3] for trn in quad.transitions)
        assert quad.scatterer.constant == qeye(m)
        assert quad.evaluate(0) == liouvillian(tensor(qeye(2), create(3) * destroy(3)),
                                               [tensor(qeye(2), create(3) * destroy(3))] +
                                               [tensor(qeye(2), destroy(3))] * m)

    _test_modes(1)
    _test_modes(2)


def test_evaluatequad_tensor_variable():
    def _test_modes(m):
        quad = make_full_quad_variable(3, m)
        quad = quad.tensor_insert(1, [2, 3])
        assert quad.modes == m
        assert quad.subdims == [2, 3]
        assert quad.hamiltonian.evaluate(2, {'a': 1, 'b': 2}) == tensor(qeye(2), 5 * create(3) * destroy(3))
        assert all(env.subdims == [2, 3] for env in quad.environment)
        assert all(trn.subdims == [2, 3] for trn in quad.transitions)
        assert quad.transitions[0].evaluate(2, {'a': 1, 'b': 2}) == tensor(qeye(2), 5 * destroy(3))
        assert quad.scatterer.constant == qeye(m)
        assert quad.evaluate(2, {'a': 1, 'b': 2}) == liouvillian(tensor(qeye(2), 5 * create(3) * destroy(3)),
                                                                 [tensor(qeye(2), 5 * create(3) * destroy(3))] +
                                                                 [tensor(qeye(2), 5 * destroy(3))] * m)

    _test_modes(1)
    _test_modes(2)


def test_evaluatequad_add():
    quad0 = make_full_quad(3, 1)
    quad1 = make_full_quad(3, 2)
    quad1.scatterer.constant = Qobj(inpt=[[1, 1], [1, 1]])
    quad = quad0 + quad1
    assert quad.subdims == [3]
    assert quad.hamiltonian.evaluate(0) == 2 * create(3) * destroy(3)
    assert quad.modes == 3
    assert quad.scatterer.constant == Qobj(inpt=[[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    assert quad.evaluate(0) == liouvillian(2 * create(3) * destroy(3),
                                           [create(3) * destroy(3)] * 2 +
                                           [destroy(3)] * 3)


def test_quad_pad():
    quad = make_full_quad(3, 2)
    quad.pad(3)
    assert quad.modes == 3
    assert quad.hamiltonian.evaluate(0) == num(3)
    assert [env.evaluate(0) for env in quad.environment] == [num(3), destroy(3), destroy(3)]
    assert [trn.evaluate(0) for trn in quad.transitions] == [destroy(3), destroy(3), qzero(3)]
    assert quad.scatterer.evaluate(0) == qeye(3)


def test_quad_permute():
    quad = make_full_quad(3, 1)
    quad.pad(2)
    quad.permute([1, 0])
    assert [env.evaluate(0) for env in quad.environment] == [num(3), destroy(3)]
    assert [trn.evaluate(0) for trn in quad.transitions] == [qzero(3), destroy(3)]


def test_quad_cascaded_mul_constant():
    quad0 = make_full_quad(2, 2)
    quad1 = make_full_quad(3, 2)

    quad2 = quad0 * quad1
    ham_expected = tensor(create(2) * destroy(2), qeye(3)) + tensor(qeye(2), create(3) * destroy(3))

    superQ = 2.j * spre((1.j / 2) * (tensor(destroy(2), create(3)) - tensor(create(2), destroy(3)))) + \
             -2.j * spost((1.j / 2) * (tensor(destroy(2), create(3)) - tensor(create(2), destroy(3)))) - \
             2 * lindblad_dissipator(tensor(destroy(2), qeye(3))) - \
             2 * lindblad_dissipator(tensor(qeye(2), destroy(3))) + \
             2 * lindblad_dissipator(tensor(destroy(2), qeye(3)) + tensor(qeye(2), destroy(3)))

    c_ops_expected = [tensor(create(2) * destroy(2), qeye(3)),
                      tensor(destroy(2), qeye(3)),
                      tensor(destroy(2), qeye(3)),
                      tensor(qeye(2), create(3) * destroy(3)),
                      tensor(qeye(2), destroy(3)),
                      tensor(qeye(2), destroy(3)),
                      superQ]

    assert quad2.hamiltonian.evaluate(0) == ham_expected
    assert [env.evaluate(0) for env in quad2.environment] == c_ops_expected
    assert [trn.evaluate(0) for trn in quad2.transitions] == [
        tensor(destroy(2), qeye(3)) + tensor(qeye(2), destroy(3)),
        tensor(destroy(2), qeye(3)) + tensor(qeye(2), destroy(3))]
    assert quad2.scatterer.evaluate(0) == qeye(2)
    assert quad2.evaluate(0) == liouvillian(ham_expected, c_ops=c_ops_expected)


def test_quad_cascaded_mul_variable():
    quad0 = make_full_quad_variable(2, 2)
    quad1 = make_full_quad_variable(2, 2)

    quad2 = quad0 * quad1
    assert quad2.hamiltonian.evaluate(2, {'a': 2, 'b': 5}) == \
           tensor(9 * create(2) * destroy(2), qeye(2)) + tensor(qeye(2), 9 * create(2) * destroy(2))
    hamQ = - 2 * (1.j / 2) * (tensor(11 * destroy(2), 11 * create(2)) - tensor(11 * create(2), 11 * destroy(2)))
    superQ = -1.j * (spre(hamQ) - spost(hamQ)) - \
             2 * lindblad_dissipator(tensor(11 * destroy(2), qeye(2))) - \
             2 * lindblad_dissipator(tensor(qeye(2), 11 * destroy(2))) + \
             2 * lindblad_dissipator(tensor(11 * destroy(2), qeye(2)) + tensor(qeye(2), 11 * destroy(2)))

    assert [env.evaluate(2, {'a': 2, 'b': 5}) for env in quad2.environment] == \
           [tensor(9 * create(2) * destroy(2), qeye(2)),
            tensor(11 * destroy(2), qeye(2)),
            tensor(11 * destroy(2), qeye(2)),
            tensor(qeye(2), 9 * create(2) * destroy(2)),
            tensor(qeye(2), 11 * destroy(2)),
            tensor(qeye(2), 11 * destroy(2)),
            superQ]
    assert [trn.evaluate(2, {'a': 2, 'b': 5}) for trn in quad2.transitions] == \
           [tensor(11 * destroy(2), qeye(2)) + tensor(qeye(2), 11 * destroy(2)),
            tensor(11 * destroy(2), qeye(2)) + tensor(qeye(2), 11 * destroy(2))]


def test_quad_cascaded_mul_variable_scattering():
    quad0 = EvaluatedQuadruple(hamiltonian=EvaluatedOperator(constant=create(2) * destroy(2)),
                               environment=[EvaluatedOperator(constant=destroy(2))] * 2,
                               transitions=[EvaluatedOperator(constant=destroy(2)),
                                            EvaluatedOperator(constant=destroy(2),
                                                              variable=[OpFuncPair(op=destroy(2),
                                                                                   func=lambda t, args:
                                                                                   args['a'] * t ** 2)])],
                               scatterer=EvaluatedOperator(constant=qeye(2),
                                                           variable=[OpFuncPair(op=create(2) + destroy(2),
                                                                                func=lambda t, args:
                                                                                args['b'] * t)]))
    quad1 = quad0 * quad0
    assert quad1.hamiltonian.evaluate(2, {'a': 1, 'b': 2}) == \
           tensor(create(2) * destroy(2), qeye(2)) + tensor(qeye(2), create(2) * destroy(2))

    hamQ = - (1.j / 2) * (tensor(destroy(2), 21 * create(2)) - tensor(create(2), 21 * destroy(2))) \
           - (1.j / 2) * (tensor(destroy(2), 5 * 9 * create(2)) - tensor(create(2), 5 * 9 * destroy(2)))
    superQ = -1.j * (spre(hamQ) - spost(hamQ)) - \
             2 * lindblad_dissipator(tensor(destroy(2), qeye(2))) - \
             2 * lindblad_dissipator(tensor(qeye(2), destroy(2))) + \
             lindblad_dissipator(tensor(qeye(2), destroy(2)) + 21 * tensor(destroy(2), qeye(2))) + \
             lindblad_dissipator(tensor(qeye(2), 5 * destroy(2)) + 9 * tensor(destroy(2), qeye(2)))

    assert [env.evaluate(2, {'a': 1, 'b': 2}) for env in quad1.environment][:-1] == \
           [tensor(destroy(2), qeye(2))] * 2 + \
           [tensor(qeye(2), destroy(2))] * 2
    assert [trn.evaluate(2, {'a': 1, 'b': 2}) for trn in quad1.transitions] == \
           [tensor(qeye(2), destroy(2)) + 21 * tensor(destroy(2), qeye(2)),
            tensor(qeye(2), 5 * destroy(2)) + 9 * tensor(destroy(2), qeye(2))]
    assert quad1.scatterer.evaluate(2, {'a': 1, 'b': 2}) == Qobj([[1, 4], [4, 1]]) * Qobj([[1, 4], [4, 1]])


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


def test_quad_cascaded_mul():
    bs = _make_time_dependent_bs()
    bs.name = 'BS0'
    op0 = bs.partial_evaluate(pi / 2, {'BS0' + d + 'a': 1})
    assert op0.evaluate(pi / 2) == Qobj([[0, 1.j], [1.j, 0]])

    bs.name = 'BS1'
    op1 = bs.partial_evaluate(pi / 2, {'BS1' + d + 'a': 2})
    assert op1.evaluate(pi / 2) == Qobj([[-1, 0], [0, -1]])

    par = {'BS0' + d + 'a': 1, 'BS1' + d + 'a': 2}
    op2 = op1 * op0
    assert op2.constant == qzero(2)
    assert op2.evaluate(pi / 2, par) == Qobj([[-1, 0], [0, -1]]) * Qobj([[0, 1.j], [1.j, 0]])

    quad0 = EvaluatedQuadruple(scatterer=op0)
    quad1 = EvaluatedQuadruple(scatterer=op1)
    quad2 = EvaluatedQuadruple(scatterer=EvaluatedOperator(Qobj([[1, 0], [0, 1]])))
    quad3 = quad0 * quad1 * quad2
    assert quad3.scatterer.evaluate(pi / 2, par) == Qobj([[-1, 0], [0, -1]]) * Qobj([[0, 1.j], [1.j, 0]])
