from ..base_processor import ProcessorBase
from ...network import DetectorGate, make_masked_source
from .photon_statistics import compute_average_photon_number, compute_intensity_correlation, compute_parity_summation
from ...time.parameters import TupleDict
from math import isclose
from typing import Union, List
from numpy import real
import matplotlib.pyplot as plt


class Distribution(TupleDict):

    def __init__(self, dictionary: dict = None, precision: int = None):
        self.precision = 6 if precision is None else precision
        self.display_precision = 4
        super().__init__(dictionary)

    def norm_function(self, value):
        return abs(value)

    def chop_function(self, value):
        return value

    @property
    def norm(self):
        return sum(self.norm_function(v) for v in self.values())

    def is_normalized(self, rtol=None, atol=None):
        return isclose(self.norm(), 1,
                       rel_tol=10 ** (-self.precision) if rtol is None else rtol,
                       abs_tol=10 ** (-self.precision) if atol is None else atol)

    def normalize(self):
        for k, v in self.items():
            self[k] = self[k] / self.norm

    def chop(self, normalize: bool = True):
        dictionary = {k: self.chop_function(v) for k, v in self.items()
                      if self.norm_function(v) > 10 ** (-self.precision + 1)}
        self.clear()
        self.update(dictionary)
        if normalize:
            self.normalize()


class ScalarDistribution(Distribution):

    def __init__(self, dictionary: dict = None, precision: int = None, type='positive'):
        self.type = type
        super().__init__(dictionary, precision)

    def __setitem__(self, key, value):
        assert isinstance(key, tuple), "Key must be a tuple"
        if self.type == 'real' or self.type == 'positive':
            value = real(value)
        if self.type == 'positive':
            if value < 0:
                value = 0
        super().__setitem__(key, value)

    def norm_function(self, value):
        return abs(value)

    def chop_function(self, value):
        return float(round(abs(value), self.precision)) if self.type == 'positive' else \
            float(round(real(value), self.precision)) if self.type == 'real' else \
                round(value, self.precision)

    @property
    def norm(self):
        return sum(self.norm_function(v) for v in self.values())

    def is_normalized(self, rtol=None, atol=None):
        return isclose(self.norm(), 1,
                       rel_tol=10 ** (-self.precision) if rtol is None else rtol,
                       abs_tol=10 ** (-self.precision) if atol is None else atol)

    def normalize(self):
        for k, v in self.items():
            self[k] = self[k] / self.norm

    @property
    def _header(self):
        return ["Number", "Probability"]

    def __str__(self):
        size = str(max(8, max(sum(len(str(i)) for i in k) + len(k) for k in self.keys())))
        format_header = "{:<" + size + "}| {:}"
        format_row = "{key:<" + size + "}| {number:.{digits}f}"
        return "\n".join([format_header.format(*self._header)] +
                         [format_row.format(key=' '.join((str(i) for i in k)), number=v, digits=self.precision - 1)
                          for k, v, in self.items()])

    def real(self):
        self.update({k: real(v) for k, v in self.items()})

    def abs(self):
        self.update({k: abs(v) for k, v in self.items()})

    def display(self):
        print(self)
        print()


class ProbabilityDistribution(ScalarDistribution):
    """
    A dictionary of photon number strings and their corresponding probabilities,
    along with some features to modify and analyze them
    """

    def __setitem__(self, key, value):
        # assert value + 10 ** (-self.precision) >= 0, "Probability must be positive"
        super().__setitem__(key, abs(value) if value >= 0 else 0.)

    def __getitem__(self, item):
        item = self._to_tuple(item)
        return super().__getitem__(item) if self._to_tuple(item) in self.keys() else 0.

    def __add__(self, other):
        pass

    def chop_function(self, value):
        return float(round(abs(value), self.precision))


class PhotonNumberDistribution(ProbabilityDistribution):
    """
    A distribution of photon number probabilities for a single mode/detector
    """

    def precision_check(self, precision):
        if self[len(self) - 1] > 10 ** (2 - precision) and len(self) > 2:
            print("Warning: truncation may be too low to achieve requested precision.")

    def gn(self, order: int):
        return round(compute_intensity_correlation(self, order), self.display_precision)

    def g2(self):
        return round(self.gn(2), self.display_precision)

    def mu(self):
        return round(compute_average_photon_number(self), self.display_precision)

    def beta(self):
        return round(1 - self[0], self.display_precision)

    def parity_sum(self):
        return round(compute_parity_summation(self), self.display_precision)

    def figures_of_merit(self):
        return {'beta': self.beta(), 'mu': self.mu(), 'g2': self.g2()}

    def display_figures(self):
        print("{:<21} | {:}".format("Figure of Merit", "Value"))
        quantities = {'Brightness': self.beta(), 'Average photon number': self.mu(), 'Intensity correlation': self.g2()}
        for k, v in quantities.items():
            print("{key:<21} | {number:.{digits}f}".format(key=k, number=v, digits=self.display_precision))
        print()

    def plot(self, label: str = '', color=None):
        plt.bar([k[0] for k in self.keys()], [v for v in self.values()], label=label, color=color)
        plt.xlabel('Number of photons, $n$')
        plt.ylabel('Photon number probability, $p(n)$')
        if label:
            plt.legend()
        return plt


class CorrelationDistribution(ScalarDistribution):
    """
    A distribution of detection probabilities for mixed detection patterns
    """

    def __setitem__(self, key, value):
        assert isinstance(key, tuple), "Key must be a tuple"
        super().__setitem__(key, value)

    def __getitem__(self, item):
        item = self._to_tuple(item)
        return super().__getitem__(item) if self._to_tuple(item) in self.keys() else 0.

    @property
    def _header(self):
        return ["Pattern", "Probability" if self.type == 'positive' else "Expectation"]

    def trace(self, position: Union[int, List[int]]):
        if isinstance(position, list):
            for p in reversed(sorted(position)):
                self.trace(p)
        else:
            pass


# Compute the photon number probabilities of a source using a single-mode Processor
def compute_photon_number_distribution(source: ProcessorBase, truncation: int, port: int = 0, parameters: dict = None):
    p = ProcessorBase(make_masked_source(source.component, port))
    p.add(0, DetectorGate(resolution=truncation))
    p.copy_conditions(source)

    distribution = PhotonNumberDistribution(precision=p.precision)
    probs = p.probs(parameters=parameters)
    distribution.update(probs)

    return distribution


class StateDistribution(Distribution):

    def norm_function(self, value):
        return abs(value.tr())

    def __getitem__(self, item):
        item = self._to_tuple(item)
        return super().__getitem__(item) if self._to_tuple(item) in self.keys() else 0. * list(self.values())[0]



class ChannelDistribution(Distribution):

    def norm_function(self, value):
        return abs(value.tr())

    def __getitem__(self, item):
        item = self._to_tuple(item)
        return super().__getitem__(item) if self._to_tuple(item) in self.keys() else 0. * list(self.values())[0]