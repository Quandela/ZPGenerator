from ...system import ScattererBase
from qutip import qeye
from typing import Union, List


class IdentityScatterer(ScattererBase):
    """
    An identity scattering matrix
    """
    def __init__(self, modes: Union[int, List[int]]):
        super().__init__(qeye(modes))
