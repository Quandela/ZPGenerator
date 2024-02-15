from zpgenerator.network.port import *
from zpgenerator.network.detector import DetectorGate
from pytest import raises
from zpgenerator.time.parameters import Parameters

d = Parameters.DELIMITER

def port_tester(port: Port):
    assert not port.is_closed
    port.close()
    assert port.is_closed
    port.open()
    assert port.is_open


def test_port():
    port = Port()
    port_tester(port)


def test_input():
    port = InputPort()
    port_tester(port)


def test_output():
    port = OutputPort()
    port_tester(port)
    assert not port.is_monitored

    det = DetectorGate(gate=lambda args: [0, args['end']], parameters={'end': 1}, name='D0')
    port.add(detector=det, bin_name='B0')
    port.add(detector=DetectorGate(gate=lambda args: [args['start'], 2], parameters={'start': 1}, name='D1'), bin_name='B1')

    assert port.uses_parameter('D0' + d + 'end')
    assert port.uses_parameter('D1' + d + 'start')

    assert port.is_monitored
    assert port.detectors['D0'] == det
    assert port.bin_names == ['B0', 'B1']
    assert [time_bin.detector for time_bin in port.time_bins] == list(port.detectors.values())
    assert port.time_bin_intervals() == [[0, 1], [1, 2]]
    assert port.time_bin_intervals(parameters={'D0' + d + 'end': 0.5, 'D1' + d + 'start': 1.5}) == [[0, 0.5], [1.5, 2]]
    with raises(Exception):
        assert port.time_bin_intervals(parameters={'D0' + d + 'end': 2, 'D1' + d + 'start': 1.5}) == [[0, 0.5], [1.5, 2]]


def test_port_layer():
    layer = PortLayer([Port(), Port(is_closed=True)])
    assert layer.modes == 2
    assert layer.open_modes == 1
    assert layer.closed_modes == 1
    assert not layer.is_closed

    layer.add(Port())
    assert layer.modes == 3
    assert layer.open_modes == 2
    assert layer.closed_modes == 1

    layer = PortLayer.make(3)
    layer.ports[1].close()
    layer.add(Port(is_closed=True))

    layer.pad(5)
    assert layer.modes == 5
    assert layer.open_modes == 2
    assert layer.closed_modes == 3


def test_input_layer():
    layer = InputLayer.make(5)
    layer.pad(7)
    assert all(isinstance(port, InputPort) for port in layer.ports)
    assert layer.open_modes == layer.modes - 2
    assert layer.closed_modes == 2


def test_output_layer():
    layer = OutputLayer.make(5)
    layer.pad(6)
    assert all(isinstance(port, OutputPort) for port in layer.ports)
    assert layer.open_modes == layer.modes - 1
    assert layer.closed_modes == 1

    det0 = DetectorGate()
    det1 = DetectorGate()
    layer.ports[1].add(detector=det0, bin_name='test')
    layer.ports[4].add(detector=det1, bin_name='test')
    det = DetectorGate(gate=[0, 1], name='D')
    array = TimeBinArray([det, DetectorGate(gate=[2, 3])], name='Array')
    layer.ports[5].add(detector=array)
    assert layer.detectors['5']['Array']['D'] == det
    assert [time_bin.detector for time_bin in layer.binned_detectors['test']] == [det0, det1]
    assert [[time_bin.mode for time_bin in time_bins] for time_bins in layer.binned_detectors.values()] == \
           [[1, 4], [5], [5]]
    assert layer.bin_names == ['test']
