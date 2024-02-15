from .parameters.parameterized_object import AParameterizedObject, ParameterizedObject
from abc import abstractmethod
from typing import Union
from math import isinf


class ATimeDomain(AParameterizedObject):
    """
    A parameterized domain of time.
    """

    @abstractmethod
    def evaluate(self, parameters: dict = None) -> list:
        """
        Evaluates the domain, including infinite times and instantaneous times.

        :param parameters: a dictionary of parameters used to evaluate the domain.
        :return: a time instant or interval, or list of instants or intervals.
        """
        pass

    @abstractmethod
    def times(self, parameters: dict = None) -> list:
        """
        A list of finite times in non-decreasing order describing important stopping points along the domain

        :param parameters: a dictionary of parameters used to evaluate the times.
        :return: a list of sorted times defining all intervals and instants, neglecting infinite values
        """
        pass


class TimeInterval(ParameterizedObject, ATimeDomain):
    """
    A parameterized interval of time defined by a function that evaluates to an interval: f(parameters) = [begin, end]
    """

    def __init__(self, interval: Union[list, callable] = None, parameters: dict = None, name: str = None):
        """
        :param interval: a constant list or a function that returns a list of two real values.
        :param parameters: a dictionary of default parameters needed to evaluate the interval function.
        :param name: a name for the object to distinguish parameters.
        """
        super().__init__(parameters=parameters, name=name)
        self.interval = [float('-inf'), float('inf')] if interval is None else interval

        if isinstance(self.interval, list) and any(isinstance(i, str) for i in self.interval):
            self._parameterize()

        self.is_callback = callable(self.interval)

        self._check_interval()

    def _check_interval(self):
        test = self.interval(self.get_parameters()) if self.is_callback else self.interval
        assert len(test) == 2, "evaluate() must return a list of two values."
        assert self._check_sorted(test), "evaluate() must return values in non-decreasing order"

    def _parameterize(self):
        begin = self.interval[0]
        end = self.interval[1]
        if isinstance(begin, str) and isinstance(end, str):
            def parameterized_interval(args: dict) -> list:
                return [args[begin], args[end]]
        elif isinstance(begin, str):
            def parameterized_interval(args: dict) -> list:
                return [args[begin], end]
        else:
            def parameterized_interval(args: dict) -> list:
                return [begin, args[end]]
        self.interval = parameterized_interval

    @staticmethod
    def _check_sorted(interval):
        return list(sorted(interval)) == interval

    def _warn_reverse(self, interval):
        if not self._check_sorted(interval):
            print("TimeInterval Warning: evaluation produced a negative interval that may cause unexpected behaviour.")
        return interval

    def evaluate(self, parameters: dict = None) -> list:
        parameters = self.get_parameters(parameters)
        return self._warn_reverse(self.interval(parameters) if self.is_callback else self.interval)

    def times(self, parameters: dict = None) -> list:
        return [t for t in self.evaluate(parameters) if not isinf(float(t))]

    @classmethod
    def source_gate(cls, pulse, parameters: dict = None, name: str = None, wait: float = 24,
                    parameter_name: str = 'decay'):
        def gate(args: dict):
            pulse_times = pulse.times(args)
            return [pulse_times[0], pulse_times[-1] + wait / args[parameter_name]]

        gate = cls(interval=gate,
                   parameters={parameter_name: 1.} | (parameters if parameters else {}),
                   name=name)
        gate.add_child(pulse)

        return gate


class TimeInstant(ParameterizedObject, ATimeDomain):
    """
    A single instance of time defined by a function that returns a list of one value:
    f(parameters=None) = [time]
    """

    def __init__(self, instant, parameters: dict = None, name: str = None):
        """
        :param instant: a constant list or a function that returns a list of one real value.
        :param parameters: a dictionary of default parameters needed to evaluate the instant function.
        :param name: a name for the object to distinguish parameters.
        """
        super().__init__(parameters=parameters, name=name)
        self.is_callback = callable(instant)
        self.instant = instant if self.is_callback or isinstance(instant, list) else [instant]
        if not self.is_callback and isinstance(self.instant[0], str):
            self._parameterize()
            self.is_callback = True
        self._check_instant()

    def _check_instant(self):
        test = self.evaluate()
        assert len(test) == 1, "domain() must return a list of one value."
        assert not isinf(float(test[0])), "an instant must be a finite value of time."

    def _parameterize(self):
        key = self.instant[0]

        def parameterized_instant(args: dict) -> list:
            return [args[key]]

        self.instant = parameterized_instant

    def evaluate(self, parameters: dict = None) -> list:
        parameters = self.get_parameters(parameters)
        return self.instant(parameters) if self.is_callback else self.instant

    def times(self, parameters: dict = None) -> list:
        return self.evaluate(parameters)


def merge_intervals(intervals: list, merge_adjacent: bool = True) -> list:
    """
    # A function that takes two lists of intervals and merges them into a new list of intervals
        Example: [[1,4], [3,6], [7,8]] --> [[1,6], [7,8]]

    :param intervals: a list of intervals.
    :param merge_adjacent: whether to merge two adjacent intervals: ex) [[1, 2], [2, 3]] --> [[1, 3]]
    :return: a merged and sorted list of intervals.
    """
    sorted_intervals = sorted(intervals)
    merged_intervals = [sorted_intervals[0]]
    for interval in sorted_intervals[1::]:
        if interval[0] <= merged_intervals[-1][1] if merge_adjacent else interval[0] < merged_intervals[-1][1]:
            if len(interval) == 2:
                if interval[1] > merged_intervals[-1][1]:
                    merged_intervals[-1][1] = interval[1]
        else:
            merged_intervals.append(interval)
    return merged_intervals


def merge_times(timelist: list) -> list:
    """
    A function that merges, flattens, deletes duplicates, and sorts lists of real values.
    Example: [[1, 2, 5, 8], [2, 4, 9]] --> [1, 2, 4, 5, 8, 9]

    :param timelist: a list of lists of times.
    :return: a single merged list of times.
    """
    return sorted(list(set([time for times in timelist for time in times])))
