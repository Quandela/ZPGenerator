from zpgenerator.time.evaluate.dirac import *
from qutip import destroy, create, num, sprepost, spre, super_tensor


def test_hamiltonian_dirac_evaluate():
    d = EvaluatedDiracOperator(hamiltonian=num(2))
    assert d.evaluate() == unitary_propagation_superoperator(num(2))


def test_channel_dirac_evaluate():
    d = EvaluatedDiracOperator(channel=sprepost(destroy(2), create(2)))
    assert d.evaluate() == sprepost(destroy(2), create(2))


def test_combo_dirac_evaluate():
    ch = sprepost(destroy(2), create(2))
    d = EvaluatedDiracOperator(hamiltonian=create(2), channel=ch)
    u = unitary_propagation_superoperator(create(2))
    assert d.evaluate() == ch * u
    assert d.evaluate(commute=True) == u * ch


def test_dirac_add():
    ch0 = sprepost(destroy(2), create(2))
    d0 = EvaluatedDiracOperator(hamiltonian=create(2), channel=ch0)

    ch1 = spre(create(2))
    d1 = EvaluatedDiracOperator(hamiltonian=2 * (create(2) + destroy(2)), channel=ch1)

    d = d1 + d0

    u = unitary_propagation_superoperator(create(2) + 2 * (create(2) + destroy(2)))
    assert d.evaluate() == ch1 * ch0 * u
    assert d.evaluate(commute=True) == u * ch1 * ch0

    d = d0 + d1

    assert d.evaluate() == ch0 * ch1 * u
    assert d.evaluate(commute=True) == u * ch0 * ch1


def test_dirac_tensor_insert():
    ch = sprepost(destroy(2), create(2))
    d = EvaluatedDiracOperator(hamiltonian=num(2), channel=ch)
    d = d.tensor_insert(1, [2, 2, 3])
    assert d.evaluate() == super_tensor(sprepost(qeye(2), qeye(2)),
                                        ch * unitary_propagation_superoperator(num(2)),
                                        sprepost(qeye(3), qeye(3)))
