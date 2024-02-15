from zpgenerator.time.operator import *
from qutip import destroy, tensor, create, sprepost, qeye
from numpy import array, pi, sqrt
from zpgenerator.time.parameters import Parameters

d = Parameters.DELIMITER

def test_operator_list():
    matrix = [[1, 2], [3, 4]]
    op = Operator(matrix)
    assert not op.is_callback
    assert not op.is_super
    assert op.evaluate() == Qobj(matrix)
    assert op.dim == 2
    assert op.subdims == [2]


def test_operator_qobj():
    op = Operator(destroy(3))
    assert not op.is_callback
    assert not op.is_super
    assert op.evaluate() == destroy(3)
    assert op.dim == 3
    assert op.subdims == [3]


def test_operator_ndarray():
    matrix = [[1, 2], [3, 4]]
    op = Operator(array(matrix))
    assert not op.is_callback
    assert not op.is_super
    assert op.evaluate() == Qobj(matrix)
    assert op.dim == 2
    assert op.subdims == [2]


def test_operator_callable_list():
    op = Operator(lambda args: [[args['a'], args['b']], [args['c'], args['d']]],
                  parameters={'a': 1, 'b': 2, 'c': 3, 'd': 4})
    assert op.is_callback
    assert not op.is_super
    assert op.evaluate() == Qobj([[1, 2], [3, 4]])
    assert op.evaluate({'a': 5}) == Qobj([[5, 2], [3, 4]])
    assert op.dim == 2
    assert op.subdims == [2]


def test_operator_callable_qobj_with_tensor():
    op = Operator(lambda args: args['a'] * tensor(create(3), destroy(2)) +
                               args['b'] * tensor(destroy(3), create(2)),
                  parameters={'a': 1, 'b': 2})
    assert op.is_callback
    assert not op.is_super
    assert op.evaluate() == tensor(create(3), destroy(2)) + 2 * tensor(destroy(3), create(2))
    assert op.evaluate({'a': 5}) == 5 * tensor(create(3), destroy(2)) + 2 * tensor(destroy(3), create(2))
    assert op.dim == 6
    assert op.subdims == [3, 2]


def test_operator_init_super():
    op = Operator(lambda args: args['eta'] * sprepost(destroy(2), create(2)),
                  parameters={'eta': 1}, name='jump')
    assert op.is_callback
    assert op.is_super
    assert op.evaluate() == sprepost(destroy(2), create(2))
    assert op.dim == 4
    assert op.subdims == [2]
    assert op.parameters == ['eta']
    assert op.parameter_tree({'eta': 2}) == {'eta': 2}


def test_operator_identity():
    assert Operator.identity(5).evaluate() == qeye(5)


def test_operator_polarized():
    pol = Operator.polarised(destroy(2), create(2))
    assert pol.evaluate() == destroy(2)
    assert pol.evaluate({'theta': pi/2}) == create(2)
    assert pol.evaluate({'theta': pi/4, 'phi': pi/2}) == (destroy(2) + 1.j * create(2))/sqrt(2)


def test_composite_operator():
    op0 = Operator(lambda args: args['a'] * create(2) * destroy(2), parameters={'a': 1})
    op1 = Operator(lambda args: args['b'] * destroy(2) * create(2), parameters={'b': 0})
    pol = Operator.polarised(destroy(2), create(2), name='pol')
    comp_op = CompositeOperator()
    comp_op.add([op0, op1])
    comp_op.add(pol)
    assert comp_op.evaluate() == create(2) * destroy(2) + destroy(2)
    assert comp_op.evaluate({'a': 4}) == 4 * create(2) * destroy(2) + destroy(2)
    assert comp_op.parameters == ['a', 'b', 'pol' + d + 'phi', 'pol' + d + 'theta']
    assert comp_op.parameter_tree() == {'_Operator': {'a': 1}, '_Operator (1)': {'b': 0}, 'pol': {'phi': 0, 'theta': 0}}


def test_nested_composite_operator():
    op0 = Operator(lambda args: args['a'] * create(3) * destroy(3), parameters={'a': 1})
    op1 = Operator.polarised(destroy(3), create(3), name='pol 1')
    comp_op1 = CompositeOperator()
    comp_op1.add([op0, op1])

    op2 = Operator(lambda args: args['b'] * destroy(3) * create(3), parameters={'b': 2})
    op3 = Operator.polarised(lambda args: args['c'] * create(3) * destroy(3), create(3),
                             parameters={'c': 0},
                             name='pol 2')
    comp_op2 = CompositeOperator()
    comp_op2.add([op2, op3])

    comp_op3 = CompositeOperator()
    comp_op3.add(comp_op1)
    comp_op3.add(comp_op2)

    assert comp_op3.evaluate() == create(3) * destroy(3) + destroy(3) + 2 * destroy(3) * create(3)
    assert comp_op3.evaluate({'a': 5, 'b': 0, 'c': 3, 'theta': pi/2}) == 5 * create(3) * destroy(3) + 2 * create(3)
    assert comp_op3.evaluate({'a': 5, 'b': 0, 'c': 3, 'pol 1' + d + 'theta': pi/2}) == \
           8 * create(3) * destroy(3) + create(3)
    assert comp_op3.parameters == ['a', 'b', 'pol 1' + d + 'phi', 'pol 1' + d + 'theta', 'pol 2' + d + 'c',
                                   'pol 2' + d + 'phi', 'pol 2' + d + 'theta']
    assert comp_op3.parameter_tree() == {'_CompositeOperator':
                                             {'_Operator': {'a': 1}, 'pol 1': {'phi': 0, 'theta': 0}},
                                         '_CompositeOperator (1)':
                                             {'_Operator': {'b': 2}, 'pol 2': {'c': 0, 'phi': 0, 'theta': 0}}}
