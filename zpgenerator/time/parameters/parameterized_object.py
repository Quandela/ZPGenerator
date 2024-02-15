from .parameter_function import CompositeParameterFunction, ParameterFunction
from .dictionary import Parameters
from abc import ABC, abstractmethod
from typing import List, Union
from functools import cache
from copy import copy
from frozendict import frozendict


class AParameterizedObject(ABC):
    """
    A class that handles initialising, forwarding, and fetching parameters.
    """

    @property
    @abstractmethod
    def parameters(self) -> List[str]:
        """
        Fetches all named parameters.
        :return: a list of parameter names.
        """
        pass

    @abstractmethod
    def parameter_tree(self, parameters: dict = None) -> dict:
        """
        Fetches all parameters.
        :param parameters: a dictionary of input parameters.
        :return: a dictionary of parameters and child parameters.
        """

    @property
    @abstractmethod
    def default_parameters(self) -> dict:
        """
        Fetches default parameters.
        :return: a dictionary of parameters, or nested dictionary of all child parameters.
        """
        pass

    @abstractmethod
    def update_default_parameters(self, parameters: dict = None):
        pass

    @property
    @abstractmethod
    def local_default_parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, parameters: Union[dict, Parameters] = None) -> Union[dict, Parameters, None]:
        """
        Determines the parameters to pass on to its dependents.
        :param parameters: a dictionary of (possibly named) parameters.
        :return: a (possibly) updated dictionary of unnamed parameters, or None if nothing was done.
        """

    @property
    @abstractmethod
    def named_parameters(self) -> list:
        """
        :return: a list of named keys modified by the object.
        """
        pass

    @abstractmethod
    def name_key(self, key: str) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: the name of the parameters.
        """
        pass

    @name.setter
    @abstractmethod
    def name(self, name) -> str:
        pass

    @property
    @abstractmethod
    def default_name(self) -> str:
        """
        :return: the default name of the parameters.
        """
        pass

    @abstractmethod
    def uses_parameter(self, name: str) -> bool:
        pass


class ChildParameterizedObject:
    """
    A list containing objects or lists of objects so that the contained objects are mutable
    """

    def __init__(self, children: List[Union[AParameterizedObject, List[AParameterizedObject]]]):
        self.children = [] if children is None else children

    def __iter__(self):
        itr = []
        for child in self.children:
            if isinstance(child, list):
                itr += child
            else:
                itr.append(child)
        return iter(itr)

    def add(self, child: Union[AParameterizedObject, List[AParameterizedObject]]):
        self.children.append(child)


class ParameterizedObject(AParameterizedObject):
    """
    An object storing default parameters needed to evaluate itself or dependent objects, and a map to take an input
    dictionary of parameters onto the required parameters for the dependent objects.
    """

    def __init__(self, parameters: dict = None, name: str = None, children: list = None):
        """
        :param parameters: a dictionary of default parameters to apply before passing the parameters on.
        :param name: a name for input parameters to distinguish this object's parameters from others.
        :param children: a list of child parameters.
        """
        self._default_parameters = parameters if parameters else {}  # initialise default parameters

        self._renames = {}
        self._rename_function = CompositeParameterFunction()
        self._children = ChildParameterizedObject(children)

        self.default_name = ''.join([Parameters.DEFAULT_PREFIX, str(self.__class__.__name__)])
        self._keys = []

        self._parameter_function = CompositeParameterFunction()
        self.name = name

    def _check_keys(self):
        """
        Determines the names of all keys that will be used by itself or child objects.
        """
        self._cache_clear()
        self._keys = [child.named_parameters for child in self._children]  # initialise list of keys to watch for
        self._keys = [key for child in self._keys for key in child]  # flattening list of lists
        self._keys += list(self._default_parameters.keys())  # add any keys in default dictionary
        self._keys = list(set(self._keys))  # remove duplicates

        self._taken_keys = sorted([self._renames.get(k, k) for k in self._keys if k not in self._renames.values()])
        self._local_default_parameters = {self._renames.get(k, k): v for k, v in self._default_parameters.items()}

        self._pass_parameters = self.name is None and not self._default_parameters  # whether to pass parameters.

    @property
    def keys(self) -> List[str]:
        """
        The list of parameter names that can be taken and used by the object.
        :return:
        """
        return self._taken_keys

    @property
    def parameters(self) -> List[str]:
        self._check_keys()
        return [k for k in sorted(self.keys) if k[0] != Parameters.DEFAULT_PREFIX]

    @cache
    def uses_parameter(self, key: str):
        if key not in self._renames.keys():
            return key in self.keys or any(child.uses_parameter(Parameters.remove_name(key, child.name))
                                           for child in self._children)
        else:
            return key in self.keys

    def _cache_clear(self):
        self.uses_parameter.cache_clear()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        self._check_keys()

    @property
    def default_name(self) -> str:
        return self._default_name

    @default_name.setter
    def default_name(self, name):
        self._default_name = name

    def name_key(self, key: str):
        return key if self.name is None else Parameters.DELIMITER.join([str(self.name), key])

    @property
    def named_parameters(self) -> List[str]:
        return [self.name_key(k) for k in self.parameters]

    @property
    def default_parameters(self) -> dict:
        """
        The input parameters that will be used when evaluating this object without specifying alternative parameters.
        :return: a dictionary of parameters
        """
        self._check_keys()
        params = {}
        for i, child in enumerate(self._children):
            child_params = {self._renames.get(*[child.name_key(k)] * 2): v for k, v in child.default_parameters.items()}
            params.update(child_params)
        params.update(self._output_local_default_parameters)
        return {k: v for k, v in params.items() if k[0] != Parameters.DEFAULT_PREFIX}

    @default_parameters.setter
    def default_parameters(self, parameters):
        self._check_keys()
        self.update_default_parameters(parameters)

    def update_default_parameters(self, parameters: dict = None):
        parameters = parameters if parameters else {}
        assert all(self.uses_parameter(name) for name in parameters.keys()), "One or more parameter does not exist."
        self._update_default_parameters(parameters)

    def _update_default_parameters(self, parameters: dict = None):
        self._default_parameters.update(parameters)
        self._check_keys()

    @property
    def local_default_parameters(self):
        return self._local_default_parameters

    @property
    def _output_local_default_parameters(self):
        return self._parameter_function(Parameters(default=self.local_default_parameters)).dict

    def add_child(self, child: Union[AParameterizedObject, List[AParameterizedObject]]):
        if type(child) == list:
            for ch in child:
                self.add_child(ch)
        else:
            self._children.add(child)
            self._check_keys()

    def set_children(self, children):
        self._children = ChildParameterizedObject(children)
        self._check_keys()

    def parameter_tree(self, parameters: dict = None) -> dict:
        """
        Builds a nested dictionary of parameters that are passed to all dependent parameterized objects.
        :param parameters: a dictionary of parameters that overwrites default parameters.
        """
        self._check_keys()
        keys = self._make_unique_names()
        if self._children.children:
            child_trees = {}
            for i, child in enumerate(self._children):
                child_tree = child.parameter_tree(self.get_parameters(parameters))
                if child_tree:
                    child_trees.update({keys[i]: child_tree})
            return child_trees
        else:
            return parinit(self.default_parameters, self.get_parameters(parameters))

    def _make_unique_names(self, objects: list = None):
        keys = [[0, child.default_name if child.name is None else child.name]
                for child in (objects if objects else self._children)]
        new_keys = []
        for key in keys:
            while key in new_keys:
                key[0] += 1
            new_keys.append(key)
        return [key[1] + ' (' + str(key[0]) + ')' if key[0] != 0 else key[1] for key in new_keys]

    def set_parameters(self, parameters: Union[dict, Parameters, frozendict] = None) -> Union[dict, Parameters, None]:
        """
        Takes a dictionary of named parameters, extracts parameters associated with keys, and adds in any defaults.
        :param parameters: a dictionary of, possibly named, parameters.
        :return: a modified dictionary.
        """
        if self._pass_parameters:  # nothing to do, so we pass input parameters on
            return dict(parameters) if isinstance(parameters, frozendict) else parameters if parameters else {}

        else:  # we have something to do
            parameters = copy(parameters) if isinstance(parameters, Parameters) \
                else Parameters(default=dict(parameters)) if isinstance(parameters, frozendict) \
                else Parameters(parameters=parameters)

            if self.name:
                parameters.remove_names(self.name)  # unnames parameter keys if self has a name

            parameters.underwrite_defaults(self.local_default_parameters)  # adds in local defaults

            parameters = self._parameter_function(parameters)  # apply parameter function

            parameters.key_subset(self.uses_parameter)  # removes parameters not used by self or children

            parameters = self._rename_function(parameters)

            return parameters if parameters else {}

    def get_parameters(self, parameters: Union[dict, Parameters, frozendict] = None) -> Union[dict, None]:
        parameters = self.set_parameters(parameters if parameters else {})
        return parameters if isinstance(parameters, dict) else parameters.dict

    def create_insert_parameter_function(self, function: callable, parameters: dict = None):
        parameters = parameters if parameters else self._output_local_default_parameters
        self._parameter_function.append(ParameterFunction.overwrite(function=function, parameters=parameters))
        self._update_default_parameters(parameters)

    def create_overwrite_parameter_function(self, function: callable, parameters: dict):
        self._parameter_function.append(ParameterFunction.overwrite(function=function, parameters=parameters))
        self._update_default_parameters(parameters)

    def create_default_parameter_function(self, function: callable, parameters: dict = None):
        parameters = parameters if parameters else self._output_local_default_parameters
        self._parameter_function.append(ParameterFunction.default(function=function, parameters=parameters))
        self._update_default_parameters(parameters)

    def clear_parameter_functions(self):
        self._parameter_function = CompositeParameterFunction()

    def rename_parameter(self, parameter_name: str, new_parameter_name: str):
        rename = self._rename_function(Parameters(default={parameter_name: new_parameter_name})).dict
        self._rename_function.prepend(ParameterFunction.rename(parameter_name, new_parameter_name))
        self._renames.update(rename)


#  Updates a dictionary of default parameters with input parameters, ignoring parameters not in the default dictionary.
def parinit(default_parameters: dict, parameters: dict = None):
    parameters = {} if parameters is None else parameters
    return {k: parameters[k] if k in parameters.keys() else default_parameters[k] for k in default_parameters.keys()}
