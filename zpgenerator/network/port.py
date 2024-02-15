from ..time import merge_times
from .detector import TimeBin, TimeBinArray
from typing import List, Union


class Port:
    def __init__(self, is_closed: bool = False, name: str = None):
        self._is_closed = is_closed
        self.port_name = name

    @property
    def is_open(self):
        return not self._is_closed

    @property
    def is_closed(self):
        return self._is_closed

    def open(self):
        """Can be cascaded with another component"""
        self._is_closed = False

    def close(self):
        """Cannot be cascaded"""
        self._is_closed = True


class InputPort(Port):
    """
    An object tracking the status of an input mode to a component
    """

    def __init__(self, is_closed: bool = False):
        super().__init__(is_closed)


class OutputPort(Port, TimeBinArray):
    """
    An object tracking the status of an output mode from a component
    """

    def __init__(self, is_closed: bool = False, is_monitored: bool = False):
        Port.__init__(self, is_closed)
        TimeBinArray.__init__(self)
        self.is_monitored = is_monitored

    def _check_add(self, detector, parameters: dict = None, name: str = None):
        if detector:
            self.is_monitored = True
        return super()._check_add(detector, parameters, name)

    @property
    def is_open(self):
        return not (self._is_closed or self.is_monitored)

    def open(self):
        """Can be cascaded with another component"""
        self.__init__()


class PortLayer:
    """A layer of ports"""

    def __init__(self, ports: List[Port] = None):
        self.ports = ports if ports else []
        self._check_ports()

    def _check_ports(self):
        pass

    @property
    def modes(self) -> int:
        return len(self.ports)

    @property
    def closed_modes(self) -> int:
        return sum(port.is_closed for port in self.ports)

    @property
    def is_closed(self) -> bool:
        return all(port.is_closed for port in self.ports)

    @property
    def open_modes(self) -> int:
        return sum(port.is_open for port in self.ports)

    @property
    def is_open(self) -> bool:
        return all(port.is_open for port in self.ports)

    def _check_add(self, port: Port) -> Port:
        return port

    def add(self, port: Union[Port, List[Port]]):
        if isinstance(port, list):
            for i, prt in enumerate(port):
                self.add(prt)
        else:
            self.ports.append(self._check_add(port))
            self._check_ports()

    def pad(self, number: int, port_type=Port, is_closed: bool = True):
        self.add([port_type(is_closed=is_closed) for i in range(0, number - self.modes)])

    @classmethod
    def make(cls, port_number: int, port_type=Port):
        return cls([port_type() for i in range(0, port_number)])

    def permute(self, perm: List[int]):
        self.ports = [self.ports[i] for i in perm]

    def get_port_number(self, name: str):
        return next((i for i, port in enumerate(self.ports) if port.port_name == name), None)


class InputLayer(PortLayer):
    """A layer of input ports, handles connections for cascading components"""
    def __init__(self, ports: List[InputPort] = None):
        super().__init__(ports)

    def _check_add(self, port: InputPort) -> InputPort:
        assert isinstance(port, InputPort), "Can only add input ports to an input layer"
        return port

    def pad(self, number: int, port_type=InputPort, is_closed: bool = True):
        super().pad(number, port_type, is_closed)

    @classmethod
    def make(cls, port_number: int, port_type=None):
        return cls([(InputPort if port_type is None else port_type)() for i in range(0, port_number)])


class OutputLayer(PortLayer):
    """A layer of output ports, handles connections for cascading components, detector assignments, and bins"""
    def __init__(self, ports: List[OutputPort] = None):
        super().__init__(ports)

    def _check_add(self, port: OutputPort) -> OutputPort:
        assert isinstance(port, OutputPort), "Can only add output ports to an output layer"
        return port

    @property
    def detectors(self) -> dict:
        return {str(i): port.detectors for i, port in enumerate(self.ports) if isinstance(port, OutputPort)}

    @property
    def binned_detectors(self) -> dict:
        return TimeBinArray.bin_detectors(self.time_bins)

    def bin_all_detectors(self, bin_name: str):
        for port in self.ports:
            if isinstance(port, OutputPort):
                for time_bin in port.time_bins:
                    time_bin.name = bin_name

    @property
    def open_modes(self) -> int:
        return sum(not (port.is_closed or port.is_monitored) for port in self.ports if isinstance(port, OutputPort))

    @property
    def monitored_modes(self) -> int:
        return sum(port.is_monitored for port in self.ports if isinstance(port, OutputPort))

    @property
    def is_monitored(self) -> bool:
        return any(port.is_monitored for port in self.ports if isinstance(port, OutputPort))

    def times(self, parameters: dict = None) -> list:
        return merge_times([port.times(parameters) for port in self.ports if isinstance(port, OutputPort)])

    @property
    def bins(self):
        return len(self.binned_detectors)

    @property
    def time_bins(self) -> list:
        time_bins = []
        for i, port in enumerate(self.ports):
            if isinstance(port, OutputPort):
                for time_bin in port.time_bins:
                    if time_bin.detector:
                        time_bins.append(TimeBin(detector=time_bin.detector, name=time_bin.name, mode=i))
        return time_bins

    @property
    def bin_names(self):
        names = []
        for port in self.ports:
            if isinstance(port, OutputPort):
                names += port.bin_names
        return list(set(names))

    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        return any(port.is_time_dependent(t, parameters) for port in self.ports if isinstance(port, OutputPort))

    def pad(self, number: int, port_type=OutputPort, is_closed: bool = True):
        super().pad(number, port_type, is_closed)

    @classmethod
    def make(cls, port_number: int, port_type=None):
        return cls([(OutputPort if port_type is None else port_type)() for i in range(0, port_number)])

    def open_all(self):
        for port in self.ports:
            port.open()
