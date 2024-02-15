from ..parameters.dictionary import Parameters
from frozendict import frozendict
from functools import cache


class DefaultCache:

    def __init__(self, time_arg: bool = True):
        self.arg_num = 2 if time_arg else 1

    def __call__(self, function):

        self.function = function

        def wrapper(*args, **kwargs):

            obj = args[0]
            parameters = kwargs.get('parameters', args[self.arg_num] if len(args) > self.arg_num else None)
            if hasattr(obj, 'set_parameters'):
                parameters = obj.set_parameters(parameters)

            if isinstance(parameters, Parameters) and not parameters.user or parameters is None or parameters == {}:
                return self.cached_function(*args[:self.arg_num],
                                            frozendict(parameters.default) if parameters else None)
            else:
                return self.function(*args[:self.arg_num], parameters)

        wrapper.cache_clear = self.cache_clear

        return wrapper

    @cache
    def cached_function(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def cache_clear(self):
        self.cached_function.cache_clear()

