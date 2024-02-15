from zpgenerator.time.evaluate import *
from qutip import destroy, create, qeye, tensor, qzero, num
from numpy import cos, sin


def test_func():
    func = Func(lambda t, args: args['a'] * t ** 2)
    assert func(2, {'a': 1}) == 4
    func2 = Func(lambda t, args: args['b'] / t)
    assert func2(2, {'a': 1, 'b': 4}) == 2
    func3 = func + func2
    assert func3(2, {'a': 1, 'b': 4}) == 6
    func4 = func3 * func
    assert func4(2, {'a': 1, 'b': 4}) == 24


def test_func_parameters():
    func0 = Func(lambda t, args: args['a'] * t ** 2, args={'a': 2})
    assert func0(2) == 8
    func1 = Func(lambda t, args: args['a'] / t, args={'a': 3})
    assert func1(2) == 3/2
    func2 = func0 * func1
    assert func2(2) == 12


def test_evaluated_function():
    evafun = EvaluatedFunction(constant=2, variable=Func(lambda t, args: args['a'] * t ** 2))
    assert evafun(2, {'a': 4}) == 2 + 16
    evafun2 = EvaluatedFunction(constant=1, variable=Func(lambda t, args: args['b'] / t))
    assert evafun2(2, {'b': 2}) == 1 + 1
    evafun3 = evafun + evafun2
    assert evafun3(2, {'a': 4, 'b': 2}) == 20
    evafun4 = evafun3 * evafun
    assert evafun4(2, {'a': 4, 'b': 2}) == 20 * 18
    evafun5 = 2 * evafun3
    assert evafun5(2, {'a': 4, 'b': 2}) == 40


def test_func_mul_evalfunc():
    evafun = EvaluatedFunction(constant=2, variable=Func(lambda t, args: args['a'] * t ** 2))
    func = Func(lambda t, args: args['b'] * t)
    evafun1 = evafun * func
    assert isinstance(evafun1, EvaluatedFunction)
    assert evafun1(2, {'a': 2, 'b': 2}) == (2 + 2 * 2 ** 2) * 4
    evafun1 = func * evafun
    assert isinstance(evafun1, EvaluatedFunction)
    assert evafun1(2, {'a': 2, 'b': 2}) == (2 + 2 * 2 ** 2) * 4


def test_opfunc_init():
    pair = OpFuncPair(op=destroy(2), func=lambda t, args: args['a'] * t ** 4)
    assert pair.op == destroy(2)
    assert pair.func(2, {'a': 2}) == 2 * 2 ** 4
    assert pair.evaluate(2, {'a': 2}) == 2 * 2 ** 4 * destroy(2)


def test_opfunc_mul():
    pair0 = OpFuncPair(op=destroy(2), func=lambda t, args: args['a'] * t ** 4)
    pair1 = OpFuncPair(op=create(2), func=lambda t, args: args['b'] * t ** -2)

    pair2 = pair0 * pair1
    assert pair2.op == destroy(2) * create(2)
    assert pair2.func(2, {'a': 1, 'b': 2}) == 8

    pair3 = pair1 * pair0
    assert pair3.op == create(2) * destroy(2)
    assert pair3.func(2, {'a': 1, 'b': 2}) == 8

    pair4 = pair0 * create(2)
    assert pair4.op == destroy(2) * create(2)
    assert pair4.func(2, {'a': 1}) == 2 ** 4

    pair5 = create(2) * pair0
    assert pair5.op == create(2) * destroy(2)
    assert pair5.func(2, {'a': 1}) == 2 ** 4


def test_opfunc_mul_func():
    pair = OpFuncPair(op=destroy(2), func=lambda t, args: args['a'] * t ** 4)
    func = Func(lambda t, args: args['b'] * t ** -2)
    pair2 = pair * func
    assert isinstance(pair2, OpFuncPair)
    assert pair2(2, {'a': 2, 'b': 2}) == destroy(2) * 2 ** 4
    pair2 = func * pair
    assert isinstance(pair2, OpFuncPair)
    assert pair2(2, {'a': 2, 'b': 2}) == destroy(2) * 2 ** 4


def test_opfunc_mul_evalfunc():
    pair = OpFuncPair(op=destroy(2), func=lambda t, args: args['a'] * t ** 4)
    evalfunc = EvaluatedFunction(constant=2,
                                 variable=[Func(lambda t, args: args['b'] * t ** -2), Func(lambda t, args: t)])
    pair2 = pair * evalfunc
    assert isinstance(pair2, OpFuncPair)
    assert pair2(2, {'a': 2, 'b': 2}) == destroy(2) * (2 * 2 ** 4 * (2 + 1 / 2 + 2))
    pair2 = evalfunc * pair
    assert isinstance(pair2, OpFuncPair)
    assert pair2(2, {'a': 2, 'b': 2}) == destroy(2) * (2 * 2 ** 4 * (2 + 1 / 2 + 2))


def test_opfunc_methods():
    foo = lambda t, args: args['a'] * t ** 4
    pair = OpFuncPair(op=destroy(2), func=foo)
    assert pair.tensor_insert(0, [2, 2]).op == tensor(destroy(2), qeye(2))
    assert pair.pad_right(2).op == Qobj(inpt=[[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert pair.pad_left(2).op == Qobj(inpt=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    assert pair.spre().op == spre(destroy(2))
    assert pair.spost().op == spost(destroy(2))
    assert pair.dag().op == create(2)
    assert pair.lind().op == lindblad_dissipator(destroy(2))
    assert not pair.is_super
    assert pair.list_form() == [destroy(2), pair.func]
    assert pair.permute([1, 0]).op == create(2)
    assert pair.element(0, 0)(2, {'a': 2}) == 0
    assert pair.element(0, 1)(2, {'a': 2}) == 2 * 2 ** 4
    assert pair.element(1, 0)(2, {'a': 2}) == 0
    assert pair.element(1, 1)(2, {'a': 2}) == 0


def test_evop_init():
    op = EvaluatedOperator()
    assert op.evaluate(0) == Qobj()

    op = EvaluatedOperator(constant=destroy(2))
    assert op.constant == destroy(2)
    assert op.variable == []

    pair = OpFuncPair(op=create(2), func=lambda t, args: t ** 2)
    op = EvaluatedOperator(variable=pair)
    assert op.constant == qzero(2)
    assert op.variable == [pair]


def test_evol_add():
    foo0 = lambda t, args: t ** 2
    pair0 = OpFuncPair(op=destroy(2), func=foo0)
    op0 = EvaluatedOperator(constant=destroy(2), variable=[pair0])

    foo1 = lambda t, args: t ** -1
    pair1 = OpFuncPair(op=qeye(2), func=foo1)
    op1 = EvaluatedOperator(constant=create(2), variable=[pair1])

    op2 = op0 + op1
    assert op2.constant == destroy(2) + create(2)
    assert op2.variable[0].op == destroy(2)
    assert op2.variable[1].op == qeye(2)
    assert op2.list_form() == [destroy(2) + create(2), [destroy(2), pair0.func], [qeye(2), pair1.func]]

    op3 = op0 + qeye(2)
    assert op3.constant == destroy(2) + qeye(2)

    op4 = op0 + pair1
    assert op4.constant == destroy(2)
    assert op4.variable[1] == pair1
    assert op4.evaluate(3, {}) == destroy(2) + 3 ** 2 * destroy(2) + 1 / 3 * qeye(2)


def test_evop_mul():
    pair = OpFuncPair(op=destroy(2), func=lambda t, args: t ** 2)
    op0 = EvaluatedOperator(constant=destroy(2), variable=[pair])
    op1 = EvaluatedOperator(constant=create(2))

    op2 = op0 * op1
    assert op2.constant == destroy(2) * create(2)
    assert op2.variable[0].op == destroy(2) * create(2)
    assert op2.variable[0].func(2, {}) == 4
    assert op2.evaluate(2, {}) == 5 * destroy(2) * create(2)

    op3 = op1 * op0
    assert op3.constant == create(2) * destroy(2)
    assert op3.variable[0].op == create(2) * destroy(2)
    assert op3.variable[0].func(2, {}) == 4
    assert op3.evaluate(2, {}) == 5 * create(2) * destroy(2)

    op4 = 2 * op0
    assert op4.constant == 2 * destroy(2)
    assert op4.variable[0].op == 2 * destroy(2)
    assert op4.evaluate(2, {}) == 10 * destroy(2)


def test_evop_methods():
    foo = lambda t, args: t ** 2
    pair = OpFuncPair(op=create(2), func=foo)
    op = EvaluatedOperator(constant=destroy(2), variable=[pair])

    op1 = op.tensor_insert(0, [2, 2])
    assert op1.constant == tensor(destroy(2), qeye(2))
    assert op1.variable[0].op == tensor(create(2), qeye(2))
    assert op1.variable[0].func.func == foo

    op1 = op.concatenate(op)
    assert op1.constant == Qobj([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    assert op1.variable[0].op == Qobj([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert op1.variable[1].op == Qobj([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    op1 = op.spre()
    assert op1.constant == spre(destroy(2))
    assert op1.variable[0].op == spre(create(2))

    op1 = op.spost()
    assert op1.constant == spost(destroy(2))
    assert op1.variable[0].op == spost(create(2))

    op1 = op.dag()
    assert op1.constant == destroy(2).dag()
    assert op1.variable[0].op == create(2).dag()

    op1 = op.lind()
    assert op1.constant == lindblad_dissipator(destroy(2))
    assert op1.variable[0].op == lindblad_dissipator(create(2))

    op1 = op.permute([1, 0])
    assert op1.constant == create(2)
    assert op1.variable[0].op == destroy(2)

    assert op.element(0, 0)(2) == 0
    assert op.element(0, 1)(2) == 1
    assert op.element(1, 0)(2) == 4
    assert op.element(1, 1)(2) == 0

    foo1 = lambda t, args: t ** -1
    pair1 = OpFuncPair(op=qeye(2), func=foo1)
    op1 = EvaluatedOperator(constant=create(2), variable=[pair1])

    assert isinstance(op.element(1, 0), EvaluatedFunction)
    assert op.element(1, 0).constant == 0
    op2 = op1 * op.element(1, 0)
    assert isinstance(op2, EvaluatedOperator)
    assert op2.element(0, 0)(2) == 2
    assert op2.element(0, 1)(2) == 0
    assert op2.element(1, 0)(2) == 4
    assert op2.element(1, 1)(2) == 2

    op2 = op.element(1, 0) * op1
    assert op2.element(0, 0)(2) == 2
    assert op2.element(0, 1)(2) == 0
    assert op2.element(1, 0)(2) == 4
    assert op2.element(1, 1)(2) == 2


def test_evop_tensor():
    foo0 = lambda t, args: t ** 2
    pair0 = OpFuncPair(op=create(2), func=foo0)
    op0 = EvaluatedOperator(constant=destroy(2), variable=[pair0])

    foo1 = lambda t, args: t ** -1
    pair1 = OpFuncPair(op=qeye(3), func=foo1)
    op1 = EvaluatedOperator(constant=create(3), variable=[pair1])

    op2 = evop_tensor_flatten([op0, op1])
    assert op2.constant == tensor(destroy(2), create(3))
    assert op2.variable[0].op == tensor(destroy(2), qeye(3))
    assert op2.variable[1].op == tensor(create(2), create(3))
    assert op2.variable[2].op == tensor(create(2), qeye(3))
    assert op2.evaluate(2, {}) == tensor(destroy(2), create(3)) + (1 / 2) * tensor(destroy(2), qeye(3)) \
           + 4 * tensor(create(2), create(3)) + 2 * tensor(create(2), qeye(3))


def test_evop_mv():
    op = EvaluatedOperator(constant=Qobj([[1 / 2, 4], [1 / 4, -2]]))
    vec = [EvaluatedOperator(constant=qeye(2), variable=[OpFuncPair(op=destroy(2), func=lambda t, args: t ** 2)]),
           EvaluatedOperator(constant=qeye(2), variable=[OpFuncPair(op=create(2), func=lambda t, args: 1 / t)])]
    vec = evop_mv(op, vec)
    assert [op(2) for op in vec] == [(1 / 2 + 4) * qeye(2) + 2 * destroy(2) + 2 * create(2),
                                     (1 / 4 - 2) * qeye(2) + destroy(2) - create(2)]

    op = EvaluatedOperator(constant=Qobj([[1 / 2, 4], [1 / 4, -2]]),
                           variable=OpFuncPair(op=num(2), func=lambda t, args: args['a'] * cos(t)))
    vec = [EvaluatedOperator(constant=create(2), variable=[OpFuncPair(op=qeye(2), func=lambda t, args: t ** 2)]),
           EvaluatedOperator(constant=destroy(2), variable=[OpFuncPair(op=qeye(2), func=lambda t, args: 1 / t)])]
    vec = evop_mv(op, vec)
    assert [op(2, {'a': 3}) for op in vec] == [1 / 2 * (create(2) + 4 * qeye(2)) + 4 * (destroy(2) + 1 / 2 * qeye(2)),
                                               1 / 4 * (create(2) + 4 * qeye(2)) +
                                               (-2 + 3 * cos(2)) * (destroy(2) + 1 / 2 * qeye(2))]


def test_evop_smv():
    op = EvaluatedOperator(constant=Qobj([[1 / 2, 4], [1 / 4, -2]]),
                           variable=OpFuncPair(op=num(2), func=lambda t, args: args['a'] * sin(t)))
    vec0 = [EvaluatedOperator(constant=create(2), variable=[OpFuncPair(op=qeye(2), func=lambda t, args: t ** 2)]),
            EvaluatedOperator(constant=destroy(2), variable=[OpFuncPair(op=qeye(2), func=lambda t, args: 1 / t)])]
    vec1 = [EvaluatedOperator(constant=qeye(2), variable=[OpFuncPair(op=num(2), func=lambda t, args: t ** 3)]),
            EvaluatedOperator(constant=destroy(2), variable=[OpFuncPair(op=create(2), func=lambda t, args: t)])]
    op = evop_umv(vec1, op, vec0)
    assert op(2, {'a': 3}) == (qeye(2) + 8 * num(2)) * (1 / 2 * (create(2) + 4 * qeye(2)) +
                                                        4 * (destroy(2) + 1 / 2 * qeye(2))) + \
           (destroy(2) + 2 * create(2)) * (1 / 4 * (create(2) + 4 * qeye(2)) +
                                           (-2 + 3 * sin(2)) * (destroy(2) + 1 / 2 * qeye(2)))


def test_evop_reshape():
    op = EvaluatedOperator(constant=qeye([2, 2, 1]), variable=[OpFuncPair(op=qeye([2, 2, 1]), func=lambda t, args: t)])
    op.reshape()
    assert op.subdims == [2, 2]
    op.reshape([4])
    assert op.subdims == [4]
    assert op.variable[0].subdims == [4]
    op.reshape([4, 1, 1])
    assert op.subdims == [4, 1, 1]
    op.reshape()
    assert op.subdims == [4]
