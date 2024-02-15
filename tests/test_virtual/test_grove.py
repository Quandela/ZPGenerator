from zpgenerator.virtual.tree import *
from zpgenerator.virtual.configuration import PhysicalDetectorGate
from zpgenerator.virtual.propagator import VPropTI
from zpgenerator.network.detector import TimeBin
from zpgenerator.virtual.grove import VGrove
from qutip import create, destroy, sprepost, liouvillian
from numpy import log


sigmaX = create(2) + destroy(2)


def test_virtual_grove_unnormalised_states():
    vdetector = PhysicalDetectorGate(resolution=1, gate=[0, log(2)])

    vgrove = VGrove(initial_time=0, states=[Qobj([[0, 0], [0, 1 / 2]])])

    branch = MeasurementBranch([TimeBin(vdetector, mode=0)])
    vgrove.add_branches(time=0, branches=[branch])

    assert vgrove.trees[0].future[0].future[0].virtual_state == Qobj([[0, 0], [0, 1 / 2]])
    assert vgrove.trees[0].future[0].future[1].virtual_state == Qobj([[0, 0], [0, 1 / 2]])

    tensor = vgrove.build_tensors(0, precision=8)[0]
    assert list(tensor.tensor) == [0.5, 0.5]

    tensor.invert()

    assert list(tensor.tensor) == [0.5, 0]

    jumps = [sprepost(destroy(2), create(2))]
    vprop = VPropTI(generator=liouvillian(0 * sigmaX, c_ops=[destroy(2)]), jumps=jumps)

    vgrove.propagate(vprop, time=log(2))

    tensor = vgrove.build_tensors(0, precision=8)[0]

    assert list(tensor.tensor) == [0.25, 0.5]

    tensor.invert()

    assert list(tensor.tensor) == [0.25, 0.25]
