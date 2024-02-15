from .configuration import AVirtualDetectorGate, PhysicalDetectorGate, FourierDetectorGate
from ..network import TimeBin
from typing import List


class MeasurementBranch:
    """An object determining when and how to branch a virtual tree given a set of time bins"""

    def __init__(self, time_bins: List[TimeBin], parameters: dict = None, name: str = None):
        self.time_bins = time_bins
        self._check_detectors()

        self.parameters = parameters
        self.name = self._first_bin.name if name is None else name

        if hasattr(self._first_bin.detector, 'method'):
            method = self._first_bin.detector.method
        else:
            method = 'Threshold'

        if isinstance(self._first_bin.detector, AVirtualDetectorGate):
            self.virtual_detector = self._first_bin.detector
        elif self._first_bin.detector.resolution is None:
            self.virtual_detector = PhysicalDetectorGate(resolution=0)
        elif self._first_bin.detector.resolution == 0 and method == 'Threshold':
            self.virtual_detector = PhysicalDetectorGate(resolution=1, ignore_zero=True)
        elif self._first_bin.detector.resolution == 1 and method == 'Threshold':
            self.virtual_detector = PhysicalDetectorGate(resolution=1)
        elif method == 'Fourier' or method == 'Threshold':
            self.virtual_detector = FourierDetectorGate(resolution=self._first_bin.detector.resolution)
        else:
            assert False, "Detector method not implemented"

        intervals = [time_bin.detector.interval(self.parameters) for time_bin in self.time_bins]
        self.interval = [min([interval[0] for interval in intervals]), max([interval[1] for interval in intervals])]

    def _check_detectors(self):
        self._first_bin = self.time_bins[0]
        assert all(type(time_bin.detector) == type(self._first_bin.detector) for time_bin in self.time_bins), \
            "Detectors binned together must be of the same type."
        assert all(time_bin.detector.resolution == self._first_bin.detector.resolution
                   for time_bin in self.time_bins), "Detectors binned together must share the same detector resolution"

    @property
    def start_time(self):
        return self.interval[0]

    @property
    def end_time(self):
        return self.interval[1]

    def virtual_configurations(self):
        return self.virtual_detector.virtual_configurations
