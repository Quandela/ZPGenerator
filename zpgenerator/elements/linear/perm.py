from ...system import ScattererBase
from ...time.evaluate import permutation_qobj
from typing import List


class Permutation(ScattererBase):
    """
    A beam splitter linear-optical component
    """

    def __init__(self, perm: List[int] = None):
        """
        :param perm: a permutation
        """
        perm = perm if perm else [1, 0]
        super().__init__(permutation_qobj(perm))
        self.default_name = '_PERM'