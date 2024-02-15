from scipy.sparse import csr_matrix, block_diag
from scipy.sparse.linalg._expm_multiply import _expm_multiply_simple
from qutip import Qobj, spre, qeye, tensor, super_tensor, fock
from math import prod
from copy import deepcopy
from typing import List
from numpy import array, asarray

# A function that inserts operator op at position n in the tensor space of dims
def tensor_insert(op: Qobj, n, dims):
    assert (op.dims[0][0] if op.issuper else op.dims[0]) == (dims[n] if isinstance(dims[n], list) else [dims[n]]), \
        "Position to insert must match the dimensions of the operator"
    opvec = []
    for i in range(0, len(dims)):
        if i == n:
            opvec.append(op)
        else:
            if dims[i] != [1]:
                opvec.append(spre(qeye(dims[i]))) if op.issuper else opvec.append(qeye(dims[i]))
    new_op = super_tensor(opvec) if op.issuper else tensor(opvec)
    return new_op


def evop_tensor_flatten(matrices, *args):
    args = id_flatten(matrices)
    if any(isinstance(a, Qobj) for a in matrices):
        return tensor(*matrices)
    else:
        operators = deepcopy(matrices)
        subdims = [op.subdims for op in operators]
        new_ops = []
        for i, op in enumerate(operators):
            new_ops.append(op.tensor_insert(i, subdims))
        return prod(op for op in new_ops)


def tensor_dict(dict0: dict, dict1: dict) -> dict:
    return dict0 if dict1 == {} else {k0 + k1: tensor(v0, v1) for k0, v0 in dict0.items()
                                      for k1, v1 in dict1.items()}


def concat_diag(op0: Qobj, op1: Qobj):
    return Qobj(block_diag([op0.data, op1.data], format='csr'))


def permutation_qobj(perm: List[int]) -> Qobj:
    dim = len(perm)
    id = list(range(0, len(perm)))
    matrix = csr_matrix(([1] * dim, (id, perm)), shape=(dim, dim))
    return Qobj(inpt=matrix)


def id_flatten(objects, *default, remove: list = None):
    remove = [] if remove is None else remove
    vec = []
    for obj in objects:
        if isinstance(obj, list):
            for o in obj:
                if o not in remove:
                    vec.append(o)
        else:
            if obj not in remove:
                vec.append(obj)
    return vec


def sum_flatten(objects, default, remove: list = None):
    return sum(id_flatten(objects, default, remove=remove), default)


def sum_tensor(objects, default):
    if all(hasattr(obj, 'tensor_insert') for obj in objects):
        subdims = id_flatten([obj.subdims for obj in objects])
        return sum((obj.tensor_insert(i, subdims) for i, obj in enumerate(objects)), default)
    else:
        return prod(objects)


def expmv_scipy(time: float, m, v):
    return _expm_multiply_simple(m.data,
                                 v.data.transpose().reshape((m.shape[0], 1)),
                                 t=time).reshape(v.shape).transpose()


def expmv_qutip(time: float, m, v):
    return (time * m).expm()(v)


def expmv(time: float, m, v):
    return expmv_scipy(time, m, v)
