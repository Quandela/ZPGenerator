from ..network import AComponent, Component, ADetectorGate, TimeBin, DetectorGate
from ..system import AElement
from ..virtual import Generator, VGrove, MeasurementBranch
from typing import Union, List
from qutip import Qobj, Options, ptrace, operator_to_vector
from scipy.sparse import hstack


class ProcessorBase:
    """
    A photonic processor composed of one or more sources of light, a linear-optical circuit, and an array of detectors.
    """

    def __init__(self, component: Union[AElement, AComponent] = None):
        """
        :param component: a component to simulate
        """
        self.component = Component(component, masked=True)  # placing a mask
        # for port in self.component.input.ports:
        #     port.close()

        self._precision = 6

        self._grove = None
        self._current_time = None  # Note that current_time = None -> simulation will restart from initial conditions
        self._branches = []
        self._binned_detectors = {}

        self._probabilities = {}
        self._states = {}
        self._channels = {}

        self._initial_state = None
        self._initial_time = None
        self._final_time = None

        self._contains_unnormalised_detector = False

    def __floordiv__(self, other):
        if isinstance(other, tuple):
            if len(other) == 2:
                position = other[0]
                element = other[1]
            else:
                position = 0
                element = other[0]
        else:
            position = 0
            element = other
        self.add(position, element)
        return self

    def _check_simulate(self):
        assert self.component.dim > 1, "Processor must contain at least one quantum system."
        assert any(port.is_monitored for port in self.component.output.ports), \
            "Processor must contain at least one detector."

    @property
    def modes(self):
        return self.component.modes - self.component.output.closed_modes

    def add(self, position: int, element: Union[AElement, ADetectorGate],
            parameters: dict = None, name: str = None, bin_name: str = None):
        self.component.add(position, element, parameters, name, bin_name)

    @property
    def initial_state(self):
        return self._initial_state if self._initial_state else self.component.initial_state

    @initial_state.setter
    def initial_state(self, state: Union[Qobj, None]):
        self._current_time = None
        if state:
            assert state.dims[0] == self.component.subdims, \
                "Input state dimension must match the dimension of all quantum systems contained by the processor"
        self._initial_state = state

    @property
    def initial_time(self):
        return self._initial_time if self._initial_time is not None else self.component.initial_time

    @initial_time.setter
    def initial_time(self, t0: Union[float, int, None]):
        self._current_time = None
        self._initial_time = t0

    @property
    def final_time(self):
        return self._final_time

    @final_time.setter
    def final_time(self, t: Union[float, int]):
        self._final_time = t

    def copy_conditions(self, processor: 'ProcessorBase'):
        self.initial_state = processor.initial_state
        self.initial_time = processor.initial_time
        self.final_time = processor.final_time
        self.precision = processor.precision

    @property
    def parameters(self) -> list[str]:
        return self.component.parameters

    @property
    def default_parameters(self) -> dict:
        return self.component.default_parameters

    def update_default_parameters(self, parameters: dict):
        self.component.update_default_parameters(parameters)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision

    def _measurement_branches(self, parameters: dict = None, bin_list: list = None):
        if not self._branches:
            binned_detectors = self.component.output.binned_detectors
            bin_keys = list(binned_detectors.keys())
            if bin_list is not None:
                assert all(k < len(bin_keys) if isinstance(k, int) else k in bin_keys for k in bin_list), \
                    "One or more bins does not exist."
                binned_detectors = {k: binned_detectors.get(bin_keys[k] if isinstance(k, int) else k) for k in bin_list}
            self.binned_detectors = binned_detectors
            self._branches = [MeasurementBranch(time_bin, parameters, name)
                              for name, time_bin in self.binned_detectors.items()]
            if not self._branches:  # we simulate the natural evolution (without any measurement)
                self._branches = [MeasurementBranch(time_bins=[TimeBin(detector=DetectorGate(resolution=None))])]
        return self._branches, self.binned_detectors

    def _get_initial_time(self, times: list):
        if self.initial_time is not None:
            initial_time = self.initial_time
        elif times:
            initial_time = times[0]
        else:
            initial_time = 0
        return initial_time

    def _get_final_time(self, times: list):
        if self.final_time is not None:
            final_time = self.final_time
        elif times:
            final_time = times[-1]
        else:
            assert False, "Must specify a final time."  # could be replaced with a default convergence to steady state
        return final_time

    def _check_if_continue(self, continue_simulation: bool):
        if continue_simulation:
            assert self._grove, "No simulation to continue."
            assert self._current_time is not None, "No current time."
            return True
        else:
            self._reset_grove()
            return False

    def _reset_grove(self):
        self._current_time = None
        self._grove = []
        self._branches = []
        self.binned_detectors = {}
        self._probabilities = {}
        self._states = {}
        self._channels = {}

    def _get_states(self, basis: List[Qobj]):
        return [self.initial_state] if basis is None else basis  # a set of one or more initial states to propagate

    def _initialize_grove(self, initial_time: float, parameters: dict = None,
                          bin_list: list = None, basis: List[Qobj] = None):
        branches, binned_detectors = self._measurement_branches(parameters, bin_list)
        branch_times = sorted([branch.start_time for branch in branches])

        if self._current_time is None:  # initialize the tree(s)
            grove = VGrove(initial_time=initial_time, states=self._get_states(basis))
            branch_order = grove.initialize(time=initial_time, branches=branches)
            self._current_time = initial_time

        else:  # we are continuing a previous simulation
            grove = self._grove
            branch_order = self._branch_order
        return branch_times, branches, branch_order, binned_detectors, grove

    def _simulate_grove(self,
                        parameters: dict = None,
                        bin_list: list = None,
                        basis: List[Qobj] = None,
                        options: Options = None,
                        continue_simulation: bool = False):
        times = self.component.times(parameters)  # determine simulation stop times
        initial_time = self._get_initial_time(times)
        final_time = self._get_final_time(times)

        self._check_if_continue(continue_simulation)
        branch_times, branches, branch_order, binned_detectors, grove = \
            self._initialize_grove(initial_time, self.component.set_parameters(parameters), bin_list, basis)
        times = [self._current_time] + [t for t in times if self._current_time < t < final_time] + [final_time]

        generator = Generator(self.component, binned_detectors=binned_detectors, precision=self.precision)

        # Main propagation algorithm
        for i in range(1, len(times)):  # Propagate from initial time to final time
            t0 = times[i - 1]  # current time
            t1 = times[i]  # next stop time

            if self.component.is_dirac(t0, parameters):  # we have instant operators to apply
                grove.apply_operator(self.component.evaluate_dirac(t0, parameters))

            if t0 in branch_times:  # we begin a measurement time bin
                branch_order += grove.add_branches(t0, branches)

            # build propagator
            propagator = generator.build_propagator(t0, parameters=parameters, options=options)

            # apply propagator to all trees in the grove
            grove.propagate(propagator, t1)  # propagate to next stop time

        self._current_time = final_time
        self._grove = grove
        self._branch_order = branch_order

    def generating_points(self, parameters: dict = None,
                          basis: List[Qobj] = None, options: Options = None):
        self._simulate_grove(parameters=parameters, basis=basis, options=options)
        return [tree.get_points() for tree in self._grove]

    def generating_states(self, parameters: dict = None,
                          basis: List[Qobj] = None, options: Options = None):
        self._simulate_grove(parameters=parameters, basis=basis, options=options)
        return [tree.get_states() for tree in self._grove]

    def generating_channels(self, parameters: dict = None,
                            basis: List[Qobj] = None, options: Options = None):
        self._simulate_grove(parameters=parameters, basis=basis, options=options)
        return list(map(list, zip(*[tree.get_states() for tree in self._grove])))

    def simulate(self,
                 parameters: dict = None,
                 point_rank: int = 0,
                 bin_list: list = None,
                 dims: List[int] = None,
                 select: List[int] = None,
                 basis: list[Qobj] = None,
                 options: Options = None,
                 reset: bool = True):
        """
        :param parameters: optional parameters to modify the default parameters.
        :param point_rank: simulation rank (0 = probabilities, 1 = states, 2 = channels).
        :param bin_list: a list of integers or strings specifying which measurement bins to simulate.
        :param dims: a list of integers specifying the desired subspace dimensions of the channel.
        :param select: a list of integers specifying which subspace dimensions to trace out.
        :param basis: the orthonormal basis of initial states for channels (if rank = 2).
        :param options: options for qutip mesolve.
        :param reset: whether to continue simulation from the current time or reset from the beginning
        """
        assert self.component.is_emitter, "At least one component must be a quantum emitter."

        # Take outer product of orthonormal set to build unit elements of the density matrix
        if point_rank == 2:
            assert basis, "Please provide a state basis for a subspace to construct the effective channel"
            basis = [psi1 * psi2.dag() for psi2 in basis for psi1 in basis]

        # simulate the virtual tree
        self._simulate_grove(parameters=parameters, bin_list=bin_list, basis=basis, options=options,
                             continue_simulation=not reset)

        tensors = self._grove.build_tensors(point_rank, self.precision)
        for tensor in tensors:
            self._contains_unnormalised_detector = tensor.invert()

        results = [tensor.extract_results(dims=dims, select=select) for tensor in tensors]

        if point_rank == 0:
            self._probabilities.update(results[0])

        elif point_rank == 1:
            self._states.update(results[0])
            self._probabilities.update({k: v.tr() for k, v in results[0].items()})

        elif point_rank == 2:

            dims = basis[0].dims if dims is None else [dims, dims]
            new_dims = dims

            for k in results[0].keys():  # looping over all measurement outcomes
                ch_inpt = []
                for states in results:  # for each set of states in conditional states simulated
                    state = states[k]  # get conditional state for measurement outcome
                    state.dims = dims  # apply desired sub-dimensions
                    if select:
                        state = ptrace(state, select)  # trace out desired subspaces
                    new_dims = state.dims
                    ch_inpt.append(operator_to_vector(state).data)
                ch_inpt = hstack(ch_inpt)  # rearrange into matrix
                channel = Qobj(inpt=ch_inpt, dims=[new_dims, new_dims], type='super')  # make into super-operator
                self._channels.update({k: channel})

        else:
            return NotImplemented

    def _order_bins(self, distribution: dict):
        return {tuple(k[i] for i in self._branch_order): v for k, v in distribution.items()}

    def probs(self, parameters: dict = None, bin_list: list = None, options: Options = None, reset: bool = True):
        self.simulate(parameters=parameters, point_rank=0, bin_list=bin_list, options=options, reset=reset)
        return self._order_bins(self._probabilities)

    def conditional_states(self, parameters: dict = None, bin_list: list = None, dims: List[int] = None,
                           select: List[int] = None, options: Options = None, reset: bool = True):
        self.simulate(parameters=parameters, point_rank=1, bin_list=bin_list,
                      dims=dims, select=select, options=options, reset=reset)
        return self._order_bins(self._states)

    def conditional_channels(self, parameters: dict = None, bin_list: list = None,
                             dims: List[int] = None, select: List[int] = None, basis: List[Qobj] = None,
                             options: Options = None, reset: bool = True):
        self.simulate(parameters=parameters, point_rank=2, bin_list=bin_list,
                      dims=dims, select=select, basis=basis, options=options, reset=reset)
        return self._order_bins(self._channels)
