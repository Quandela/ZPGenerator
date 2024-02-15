from .tree import VTree
from .configuration import ParityDetectorGate, FourierDetectorGate
from qutip import Qobj, ptrace
from numpy import array, einsum, complex64, ndarray, reshape, apply_along_axis, nditer, prod
from numpy.fft import ifftn
from string import ascii_lowercase


class GeneratingTensor:

    def __init__(self, point_rank: int, virtual_tree: VTree, precision: int):
        self.size = virtual_tree.branch_number
        self.point_rank = point_rank
        self.precision = precision
        self.subdims = virtual_tree.subdims

        self.axes = ['parity' if isinstance(branch.virtual_detector, ParityDetectorGate) else
                     'fourier' if isinstance(branch.virtual_detector, FourierDetectorGate) else
                     'threshold' for branch in virtual_tree.branches]

        if self.point_rank == 0:
            self.tensor = virtual_tree.build_probability_tensor()
        else:
            self.tensor = virtual_tree.build_state_tensor()

    def parity_inverse(self):
        self.tensor = self.tensor

    def fourier_inverse(self):
        self.tensor = ifftn(array(self.tensor), axes=tuple(i for i in range(0, self.size) if self.axes[i] == 'fourier'))

    @staticmethod
    def _axis_threshold_inverse(tensor: ndarray, ax: int):
        mat = array([[1., 0.], [-1., 1.]], dtype=complex64)  # define the threshold transformation
        indices = list(ascii_lowercase)[0:1 + tensor.ndim]
        sum_index = indices.pop(ax + 1)
        ein_str = ','.join([sum_index + indices[ax], ''.join(indices)])
        return einsum(ein_str, mat, tensor)

    def threshold_inverse(self):
        tensor = self.tensor
        shape = tensor.shape
        contains_unnormalised_detector = False
        for i, axis in enumerate(self.axes):
            if axis == 'threshold':
                if shape[i] == 2:
                    tensor = self._axis_threshold_inverse(tensor, i)
                else:
                    contains_unnormalised_detector = True
        self.tensor = tensor
        return contains_unnormalised_detector

    def invert(self):
        contains_unnormalised_detector = False
        if 'parity' in self.axes:
            self.parity_inverse()
            contains_unnormalised_detector = True
        if 'fourier' in self.axes:
            self.fourier_inverse()
        if 'threshold' in self.axes:
            contains_unnormalised_detector = contains_unnormalised_detector or self.threshold_inverse()
        self.reshape_states()
        return contains_unnormalised_detector

    def reshape_states(self):
        if self.point_rank != 0:
            dim = prod(self.subdims)
            self.tensor = apply_along_axis(lambda subarray: _DummyState(state=Qobj(inpt=reshape(subarray, (dim, dim)),
                                                                                   dims=[self.subdims, self.subdims])),
                                           axis=-1,
                                           arr=reshape(self.tensor, self.tensor.shape[0:-2] + tuple([-1])))

    def _rearrange_key(self, key: tuple, perm: list = None):
        return tuple(key[i] for i in perm) if perm else key

    def _rearrange_keys(self, results: dict, perm: list = None):
        return {self._rearrange_key(k, perm): v for k, v in results.items()} if perm else results

    def extract_results(self, dims: list = None, select: list = None, perm: list = None):
        results = self._get_results()

        if self.point_rank != 0:
            results = {self._rearrange_key(k, perm): v.state for k, v in results.items()}
            if self.point_rank == 1:
                results = self.ptrace(results, dims=dims, select=select)
        else:
            results = self._rearrange_keys(results, perm)

        return results

    def _get_results(self):
        results = {}
        itr = nditer(self.tensor, flags=['multi_index', 'refs_ok'])
        for pr in itr:
            results.update({itr.multi_index: pr[()]})
        if 'parity' in self.axes:
            results = {self._relabel_parity(k): v for k, v in results.items()}
        return results

    def _relabel_parity(self, key: tuple):
        return tuple(k if self.axes[i] != 'parity' else 'p' for i, k in enumerate(key))


    @staticmethod
    def ptrace(results: dict, dims: list = None, select: list = None):
        if dims or select:
            for k, v in results.items():
                if dims:
                    v.dims = [dims, dims]
                if select:
                    results[k] = ptrace(v, select)

        return results


class _DummyState:
    def __init__(self, state: Qobj):
        self.state = state

