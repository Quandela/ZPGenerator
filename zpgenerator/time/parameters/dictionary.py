from collections import UserDict

class TupleDict(UserDict):

    def __init__(self, dictionary: dict = None):
        super().__init__()
        if dictionary:
            dictionary = {self._to_tuple(k): v for k, v in dictionary.items()}
            self.update(dictionary)

    def __getitem__(self, item):
        return super().__getitem__(self._to_tuple(item))

    def _to_tuple(self, item):
        return item if isinstance(item, tuple) else tuple([item])


class Parameters:

    CACHE_PREFIX = '$'
    DEFAULT_PREFIX = '_'
    DELIMITER = '/'
    WILDCARD = '*'

    def __init__(self, default: dict = None, parameters: dict = None):
        self.default = default if default else {}
        self.user = {}
        self._p = len(self.DEFAULT_PREFIX)
        self._c = len(self.CACHE_PREFIX)
        if parameters:
            self.add(parameters)

    @property
    def dict(self):
        return self.default | self.user

    def __getitem__(self, item):
        return self.dict[item]

    def add(self, parameters: dict):
        for name, value in parameters.items():
            is_default, name = self._is_default(name)
            if is_default:
                self.default.update({name: value})
            else:
                self.user.update({name: value})

    def _is_default(self, name: str):
        if name[:self._p] == self.DEFAULT_PREFIX:
            return True, name[self._p:]
        elif name[:self._c] == self.CACHE_PREFIX:
            return True, name[self._c:]
        else:
            return False, name

    @classmethod
    def remove_name(cls, key: str, name: str) -> str:
        name_list = key.split(cls.DELIMITER)
        if name_list[0] == name or name_list[0] == cls.WILDCARD:
            return cls.DELIMITER.join(name_list[1:])
        else:
            return key

    def remove_names(self, name: str):
        self.key_function(lambda key: self.remove_name(key, name))

    def key_function(self, function: callable):
        self.default = {function(k): v for k, v in self.default.items()}
        self.user = {function(k): v for k, v in self.user.items()}

    def key_subset(self, function: callable):
        self.default = {k: v for k, v in self.default.items() if function(k)}
        self.user = {k: v for k, v in self.user.items() if function(k)}

    def underwrite_defaults(self, parameters: dict):
        self.default = parameters | self.default
