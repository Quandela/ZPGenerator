from zpgenerator.virtual.tree import *
from zpgenerator.virtual.configuration import PhysicalDetectorGate
from zpgenerator.virtual.propagator import VPropHTD, VPropTI
from zpgenerator.network.detector import TimeBin
from zpgenerator.time import Func
from qutip import fock, create, destroy, sprepost, fidelity, liouvillian
from math import isclose
from numpy import pi, exp, sqrt, log


sigmaX = create(2) + destroy(2)


def gaussian(t, args):
    return pi * exp(-(t - args['delay']) ** 2 / (2 * args['width'] ** 2)) / sqrt(2 * pi * args['width'] ** 2)


def test_physical_detector_in_virtual_tree():
    vdetector = PhysicalDetectorGate(resolution=5, efficiency=0.5, gate=[0, 5])
    assert vdetector.coupling(0) == 0.5
    assert vdetector.virtual_configurations == [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

    vtree = VTree(initial_state=VState(state=fock(2), time=-4))

    branch = MeasurementBranch([TimeBin(vdetector, mode=0)])
    vtree.add_branch(branch)

    vprop = VPropHTD(hamiltonian=[0.1 * create(2) * destroy(2), [sigmaX / 2,
                                                                 Func(gaussian, args={'delay': 0, 'width': 1})]],
                     collapse_operators=[destroy(2)],
                     jumps=[sprepost(destroy(2), create(2)), sprepost(destroy(2), create(2))])

    vtree.propagate(vprop, 4)

    expected_points = [0.160164, 0.306841, 0.463903, 0.631608, 0.810218, 1.000000]
    assert all(isclose(point, expected_points[i], abs_tol=1e-5) for i, point in enumerate(vtree.get_points()))

    expected_fidelities = [0.245431, 0.468706, 0.610876, 0.731182, 0.840158, 0.9421293]
    assert all(isclose(fidelity(state, Qobj([[1, 1], [1, -1]])), expected_fidelities[i], abs_tol=1e-5)
               for i, state in enumerate(vtree.get_states()))

    vtree.apply_operator(sigmaX)
    expected_fidelities = [0.0, 0.123157, 0.148035, 0.162469, 0.173003, 0.181629]
    assert all(isclose(fidelity(state, Qobj([[1, 1], [1, -1]])), expected_fidelities[i], abs_tol=1e-5)
               for i, state in enumerate(vtree.get_states()))

    vtree.add_branch(branch)
    vtree.propagate(vprop, 5)
    assert len(vtree.get_points()) == 36

def test_virtual_tree_unnormalised_states():
    vdetector = PhysicalDetectorGate(resolution=1, gate=[0, log(2)])

    vtree = VTree(initial_state=VState(state=Qobj([[0, 0], [0, 1 / 2]]), time=0))

    branch = MeasurementBranch([TimeBin(vdetector, mode=0)])
    vtree.add_branch(branch)

    assert vtree.future[0].future[0].virtual_state == Qobj([[0, 0], [0, 1 / 2]])
    assert vtree.future[0].future[1].virtual_state == Qobj([[0, 0], [0, 1 / 2]])


    jumps = [sprepost(destroy(2), create(2))]
    vprop = VPropTI(generator=liouvillian(0 * sigmaX, c_ops=[destroy(2)]), jumps=jumps)

    assert vtree.get_points() == [0.5, 0.5]

    vtree.propagate(vprop, log(2))
    assert vtree.get_points() == [0.25, 0.5]
    assert vtree.get_states() == [Qobj([[0, 0], [0, 1 / 4]]), Qobj([[1 / 4, 0], [0, 1 / 4]])]
