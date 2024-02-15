from .parameterized_object import AParameterizedObject, ParameterizedObject, ChildParameterizedObject
from abc import abstractmethod
from typing import Union, List
from copy import deepcopy


class AParameterizedCollection(AParameterizedObject):
    """
    A collection of objects with parameters.
    """

    @abstractmethod
    def _check_objects(self):
        pass

    @abstractmethod
    def _check_add(self, obj, parameters: dict = None, name: str = None):
        pass

    @property
    @abstractmethod
    def objects(self):
        pass

    @abstractmethod
    def add(self, objects, parameters: dict = None, name: str = None):
        pass


class ParameterizedCollection(ParameterizedObject, AParameterizedCollection):
    """
    """

    def __init__(self,
                 objects: Union[ParameterizedObject, List[ParameterizedObject]] = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        self._objects = []
        super().__init__(parameters=parameters, name=name, children=self._objects)

        self._object_types = [ParameterizedObject, ParameterizedCollection] if types is None else types

        objects = [] if objects is None else objects
        self.add(objects)

    def _check_objects(self):
        self._check_keys()

    def _check_add(self, obj, parameters: dict = None, name: str = None):
        if isinstance(obj, AParameterizedObject):
            if parameters or name:
                obj = deepcopy(obj)
                obj.update_default_parameters(parameters)
                obj.name = name
        return obj

    @property
    def objects(self) -> list:
        return self._objects

    @objects.setter
    def objects(self, objects):
        objects = objects if isinstance(objects, list) else [objects]
        self._objects = objects
        self._children = ChildParameterizedObject(children=self._objects)
        self._check_objects()

    def add(self, objects, parameters: dict = None, name: str = None):
        """
        Adds an object or list of objects to this collection
        :param name:
        :param parameters:
        :param objects: an object or list of objects to add
        """
        if isinstance(objects, list):
            for obj in objects:
                self.add(obj, parameters, name)
        elif any(isinstance(objects, typ) for typ in self._object_types):
            self._add(objects, parameters, name)
            self._check_objects()
        else:
            assert False, "Cannot add object of type {t}.".format(t=type(objects))

    def _add(self, objects, parameters: dict = None, name: str = None):
        self._objects.append(self._check_add(objects, parameters, name))

    def __add__(self, other):
        """
        Merges two collections, overwriting the second one's name
        :param other: another collection of objects
        :return: a new collection of objects
        """
        return self.__class__(self._objects + other.objects,
                              parameters=self.default_parameters | other.default_parameters,
                              name=self.name)
