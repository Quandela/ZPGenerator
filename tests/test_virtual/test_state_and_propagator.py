from zpgenerator.virtual.propagator import *
from zpgenerator.virtual.state import VState
from zpgenerator.time import OpFuncPair, Func
from qutip import fock, create, destroy, num, fidelity, sprepost
from numpy import pi, exp, sqrt, log
from math import isclose

sigmaX = create(2) + destroy(2)


def gaussian(t, args):
    return pi * exp(-(t - args['delay']) ** 2 / (2 * args['width'] ** 2)) / sqrt(2 * pi * args['width'] ** 2)


def test_state_init():
    vstate = VState(state=fock(2))
    assert vstate.time == 0


def test_state_manipulate_time_independent():
    vstate = VState(state=fock(2, 0), time=0)

    assert vstate.apply_operator(sigmaX) == fock(2, 1) * fock(2, 1).dag()

    vprop = VPropTI(generator=sigmaX / 2)
    vstate.propagate(propagator=vprop, t=pi / 2)
    assert vstate == (fock(2, 0) + 1.j * fock(2, 1)) * (fock(2, 0) + 1.j * fock(2, 1)).dag() / 2


def test_state_manipulate_herm_time_dep():
    vstate = VState(state=fock(2, 0), time=0)
    func = Func(gaussian, args={'delay': 4, 'width': 0.5})
    vprop = VPropHTD(hamiltonian=[0 * num(2), [sigmaX / 2, func]])
    vstate.propagate(propagator=vprop, t=8, tlist=[0, 4, 8])
    assert isclose(abs(vstate[1][0, 0] * 1.j), 1.)


def test_state_manipulate_nonherm_time_dep():
    vstate = VState(state=fock(2, 0), time=0)
    vprop = VPropNHTD(generator=EvaluatedOperator(constant=liouvillian(H=create(2) * destroy(2),
                                                                       c_ops=[destroy(2)]),
                                                  variable=[OpFuncPair(op=liouvillian(H=sigmaX / 2,
                                                                                      c_ops=[create(2) * destroy(2)]),
                                                                       func=Func(gaussian, args={'delay': 0.1, 'width': 0.01}))]))

    vstate.propagate(propagator=vprop, t=0.2)
    assert isclose(fidelity(vstate, Qobj([[0.34553939+0.j, -0.00580815+0.02835213j],
                                          [0.00580815-0.02835213j, 0.65446061+0.j]])), 1, abs_tol=1e-7)


def test_state_jumps():
    vstate = VState(state=fock(2, 1), time=0, virtual_configuration=[1])
    jumps = [sprepost(destroy(2), create(2))]
    vprop = VPropTI(generator=liouvillian(0 * sigmaX, c_ops=[destroy(2)]), jumps=jumps)
    vstate.propagate(propagator=vprop, t=2)
    assert vstate[0][0, 0] == 0

    vstate = VState(state=fock(2, 1), time=0, virtual_configuration=[0.5])
    vprop = VPropTI(generator=liouvillian(sigmaX / 2, c_ops=[destroy(2)]), jumps=jumps)
    vstate.propagate(propagator=vprop, t=2)
    assert isclose(fidelity(vstate, vstate), 0.6228716916889016)


def test_state_jumps_unnormalised():
    vstate = VState(state=Qobj([[0, 0], [0, 0.5]]), time=0, virtual_configuration=[1])
    jumps = [sprepost(destroy(2), create(2))]
    vprop = VPropTI(generator=liouvillian(0 * sigmaX, c_ops=[destroy(2)]), jumps=jumps)
    vstate.propagate(propagator=vprop, t=log(2))
    assert vstate == Qobj([[0, 0], [0, 0.25]])

