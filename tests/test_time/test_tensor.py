from zpgenerator.time.evaluate.tensor import *
from qutip import destroy, tensor, super_tensor,qeye, spre, create


def test_tensor_insert():
    op = destroy(3)
    assert tensor_insert(op, 1, [4, 3, 3]) == tensor(qeye(4), op, qeye(3))

    op = spre(destroy(3))
    assert tensor_insert(op, 1, [4, 3, 3]) == super_tensor(spre(qeye(4)), op, spre(qeye(3)))

    op = tensor(destroy(2), destroy(3))
    assert tensor_insert(op, 1, [2, [2, 3], 3]) == tensor(qeye(2), op, qeye(3))
    assert tensor_insert(op, 2, [2, [2, 2], [2, 3]]) == tensor(qeye(2), qeye([2, 2]), op)


def test_permute():
    assert permutation_qobj([1, 0]) * destroy(2) == create(2) * destroy(2)
    assert destroy(2) * permutation_qobj([1, 0]) == destroy(2) * create(2)


def test_concat_diag():
    mat0 = qeye(2)
    mat1 = destroy(2)
    mat2 = concat_diag(mat0, mat1)
    assert mat2 == Qobj([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])


def test_permutation_qobj():
    perm = permutation_qobj([0, 2, 1])
    assert perm == Qobj([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
