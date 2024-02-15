from abc import abstractmethod
from ..time import ATimeFunction, TimeInterval, TimeFunction, TimeIntervalFunction, merge_intervals, Func, merge_times
from ..time.parameters import ParameterizedObject, ParameterizedCollection, AParameterizedObject
from typing import Union, List
from inspect import signature
from copy import copy, deepcopy


class ADetectorGate(AParameterizedObject):
    """
    A detector with an efficiency, resolution, and a coupling function describing the gate.
    """

    @property
    @abstractmethod
    def resolution(self) -> int:
        """
        :return: the number of photons or (configurations - 1) that the detector can resolve.
        """
        pass

    @property
    @abstractmethod
    def gate(self):
        pass

    @abstractmethod
    def interval(self, parameters: dict = None):
        pass

    @property
    @abstractmethod
    def efficiency(self):
        """
        :return: a value between 0 and 1 or function returning such a value.
        """
        pass

    @abstractmethod
    def coupling(self, t: float, parameters: dict = None) -> float:
        """
        :param t: time
        :param parameters: a dictionary of optional arguments used to modify the gate function.
        :return: the coupling function between the detector and the monitored mode at time t.
        """
        pass

    @abstractmethod
    def coupling_function(self, t: float, parameters: dict = None) -> Union[any, Func]:
        pass

    @abstractmethod
    def times(self, parameters: dict = None) -> list:
        pass


class DetectorGate(ParameterizedObject, ADetectorGate):
    """
    A detector that monitors a single mode over a single interval of time.
    """

    def __init__(self,
                 resolution: Union[int, None] = 1,
                 efficiency: Union[int, float] = 1.,
                 gate: Union[ATimeFunction, TimeInterval, list, callable] = None,
                 method: str = 'Threshold',
                 parameters: dict = None,
                 name: str = None):
        """
        :param resolution: the number of photons that the detector can resolve.
        :param efficiency: a constant physical efficiency multiplier for the detector.
        :param gate: an optional function or interval describing the detector gate and efficiency.
        """
        self._efficiency = efficiency
        self.method = method

        if isinstance(gate, list):
            gate = TimeInterval(interval=gate, parameters=parameters)
        elif callable(gate):
            if 't' in signature(gate).parameters:
                gate = TimeFunction(value=gate, parameters=parameters)
            else:
                gate = TimeInterval(interval=gate, parameters=parameters)

        self._gate = TimeIntervalFunction(value=1, interval=gate)

        self._resolution = resolution

        super().__init__(name=name, children=[self._gate])

    @property
    def resolution(self) -> int:
        return self._resolution

    @property
    def gate(self):
        return self._gate

    def times(self, parameters: dict = None) -> list:
        parameters = self.set_parameters(parameters)
        return self.gate.times(parameters)

    def interval(self, parameters: dict = None):
        return self.gate.support(parameters=self.set_parameters(parameters))

    @property
    def efficiency(self) -> float:
        return self._efficiency

    def coupling(self, t: float, parameters: dict = None) -> float:
        return self.efficiency * self.gate.evaluate(t, parameters)

    def coupling_function(self, t: float, parameters: dict = None) -> Union[any, Func]:
        parameters = self.set_parameters(parameters)
        if self.gate.is_time_dependent(t, parameters):
            return self.efficiency * Func(self.gate.evaluate, args=parameters)
        else:
            return self.efficiency * self.gate.evaluate(t, parameters)


class TimeBin:
    """A bin name, a mode, and a detector gate"""
    def __init__(self, detector: ADetectorGate, name: str = None, mode: int = None):
        self.detector = detector
        self.name = name
        self.mode = mode


class TimeBinArray(ParameterizedCollection):
    """
    An array of time bin detectors.
    """

    def __init__(self, detectors: List[Union[ADetectorGate]] = None, parameters: dict = None, name: str = None):
        """
        :param detectors: the list of time bin detectors.
        """
        self.time_bins = []
        self._added_bin_name = None
        super().__init__(objects=detectors, parameters=parameters, name=name, types=[ADetectorGate, TimeBin, TimeBinArray])

    def _check_objects(self, parameters: dict = None):
        self.time_bin_intervals(parameters)
        binned_detectors = self.binned_detectors
        for time_bins in binned_detectors.values():
            assert all(type(time_bin.detector) == type(time_bins[0].detector) for time_bin in time_bins), \
                "Detectors binned together must be of the same type."
            assert all(time_bin.detector.resolution == time_bins[0].detector.resolution for time_bin in time_bins), \
                "Detectors binned together must share the same detector resolution"
        self._check_keys()

    def add(self, detector, parameters: dict = None, name: str = None, bin_name: str = None):
        self._added_bin_name = bin_name
        super().add(detector, parameters, name)

    def _check_add(self, detector, parameters: dict = None, name: str = None):
        if isinstance(detector, TimeBin):
            if self._added_bin_name:
                detector = copy(detector)
                detector.name = self._added_bin_name
            self.time_bins.append(detector)
            detector = detector.detector
        elif isinstance(detector, ADetectorGate):
            self.time_bins.append(TimeBin(detector, name=self._added_bin_name))
        elif isinstance(detector, TimeBinArray):
            if self._added_bin_name:
                detector = deepcopy(detector)
                for time_bin in detector.time_bins:
                    time_bin.name = self._added_bin_name
            self.time_bins += detector.time_bins
        return super()._check_add(detector, parameters, name)

    @property
    def bin_names(self):
        return [time_bin.name for time_bin in self.time_bins if time_bin.name is not None]

    def time_bin_intervals(self, parameters: dict = None):
        parameters = self.set_parameters(parameters)
        time_bin_intervals = []
        for detector in self._objects:
            if isinstance(detector, ADetectorGate):
                time_bin_intervals.append(detector.interval(parameters=parameters))
            else:
                for interval in detector.time_bin_intervals(parameters=parameters):
                    time_bin_intervals.append(interval)
        if time_bin_intervals:
            assert len(time_bin_intervals) == len(merge_intervals(time_bin_intervals, merge_adjacent=False)), \
                "Time bins cannot overlap."
        return time_bin_intervals

    @property
    def detectors(self) -> dict:
        keys = self._make_unique_names()
        return {keys[i]: det if isinstance(det, DetectorGate) else det.detectors for i, det in enumerate(self._objects)}

    @property
    def bins(self):
        return len(self._objects)

    @property
    def binned_detectors(self) -> dict:
        return self.bin_detectors(self.time_bins)

    @staticmethod
    def bin_detectors(time_bins: List[TimeBin]):
        counter = 0
        binned_detectors = {}
        for time_bin in time_bins:
            if time_bin.name is None:
                bin_name = '_bin' if counter == 0 else '_bin (' + str(counter) + ')'
                counter += 1
            else:
                bin_name = str(time_bin.name)

            if bin_name in binned_detectors.keys():
                binned_detectors[bin_name].append(time_bin)
            else:
                binned_detectors[bin_name] = [time_bin]

        return binned_detectors

    def coupling(self, t: float, parameters: dict = None) -> float:
        parameters = self.set_parameters(parameters)
        return sum(detector.coupling(t, parameters) for detector in self._objects)

    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        parameters = self.set_parameters(parameters)
        return any(detector.gate.is_time_dependent(t, parameters) for detector in self._objects)

    def times(self, parameters: dict = None) -> list:
        parameters = self.set_parameters(parameters)
        return merge_times([detector.gate.times(parameters) for detector in self._objects])
