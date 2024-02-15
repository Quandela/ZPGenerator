from zpgenerator.network.detector import *


def test_detector_init_empty():
    det = DetectorGate()
    assert det.resolution == 1
    assert det.efficiency == 1
    assert det.coupling(0) == 1


def test_detector_init_complex():
    det = DetectorGate(resolution=2,
                       efficiency=0.5,
                       gate=lambda args: [args['start'], args['end']],
                       parameters={'eta': 1, 'start': 0, 'end': 2},
                       name='D0')

    assert det.resolution == 2
    assert det.efficiency == 0.5
    assert det.coupling(2) == 0
    assert det.coupling(1) == 0.5
    assert det.coupling(3, {'eta': 0.5, 'end': 4}) == 0.5
