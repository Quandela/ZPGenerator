from typing import List
from .dictionary import Parameters


class ParameterFunction:
    """
    An object that takes a dictionary of parameters and outputs a new dictionary of parameters with some new parameters
    computed from one or more parameters from the input dictionary.
    """

    def __init__(self, function: callable, parameters: dict = None, input_keys: list = None, output_keys: list = None):
        """
        :param function: a function of the form function(args: dict) -> dict
        :param parameters: a dictionary of default parameters used by the parameter function
        """
        self._function = function
        self.parameters = parameters if parameters else {}
        self.input_keys = input_keys if input_keys else list(self.parameters.keys())
        self.output_keys = output_keys if output_keys else list(self._function(self.parameters).keys())

    def function(self, parameters: Parameters):
        """
        Takes a Parameters object, applies the function self._function to the merged dictionary.
        Sorts output keys into default or user.
        :param parameters: a Parameters object
        :return: a modified Parameters object
        """

        parameter_dict = parameters.dict  # a merged dictionary default | user

        if all(key in parameter_dict.keys() for key in self.input_keys):  # if we have all the inputs
            parameter_dict = self._function(parameter_dict)  # apply function

            new_parameters = Parameters()

            # if none of the keys used by the function were user-specified, then the outputs are default as well
            if not any(key in parameters.user.keys() for key in self.input_keys):
                for key in self.output_keys:
                    new_parameters.default.update({key: parameter_dict.pop(key)})
            else:
                for key in self.output_keys:
                    new_parameters.user.update({key: parameter_dict.pop(key)})

            # any user-defined parameters are sorted back into the user-defined category for the output
            for key, value in parameter_dict.items():
                if key in parameters.user.keys():
                    new_parameters.user.update({key: value})
                else:
                    new_parameters.default.update({key: value})

            return new_parameters
        else:
            return parameters

    def __call__(self, parameters: Parameters) -> Parameters:
        return self.function(parameters)

    @classmethod
    def rename(cls, name: str, new_name: str):
        return cls(lambda args: {name if k == new_name else k: v for k, v in args.items() if k != name},
                   input_keys=[new_name], output_keys=[name])

    @classmethod
    def overwrite(cls, function: callable, parameters: dict = None):
        func = lambda args: args | function(args)
        return cls(func, parameters=parameters, input_keys=list(parameters.keys()),
                   output_keys=list(function(parameters).keys()))

    @classmethod
    def default(cls, function: callable, parameters: dict = None):
        func = lambda args: function(args) | args
        return cls(func, parameters=parameters, input_keys=list(parameters.keys()),
                   output_keys=list(function(parameters).keys()))


class CompositeParameterFunction:
    def __init__(self, functions: List[ParameterFunction] = None):
        self.functions = functions if functions else []

    def __call__(self, parameters: Parameters) -> Parameters:
        if self.functions:
            output_dict = self.functions[0](parameters)
            for function in self.functions[1:]:
                output_dict = function(output_dict)
            return output_dict
        else:
            return parameters

    def append(self, other: ParameterFunction):
        self.functions.append(other)

    def prepend(self, other: ParameterFunction):
        self.functions = [other] + self.functions
