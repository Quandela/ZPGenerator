from ..switches.gate import Gate
from ..losses.uniform import UniformLoss
from ...network import Component
from ...time import TimeInterval, Lifetime
from ...system import EmitterBase
from ...misc.display import Display
from ...simulate import Processor
from typing import Union, List
from qutip import Options


class SourceComponent(Component):
    """ A component with default closed inputs and one or more open outputs.
    Contains additional methods to analyze the quality of emission."""
    def __init__(self,
                 emitter: EmitterBase = None,
                 parameters: dict = None,
                 name: str = None):

        super().__init__(elements=emitter, parameters=parameters, name=name)
        self._quality_processor = None

    @property
    def quality(self):
        self._make_processor()
        return self._quality_processor.quality

    def _make_processor(self):
        if not self._quality_processor:
            self._quality_processor = Processor() // self

    def display(self):
        Display(self).display()

    def photon_statistics(self,
                          port: Union[int, str] = None,
                          parameters: dict = None,
                          truncation: int = None):
        """
        Computes the photon statistics of light emitted from a port by first computing the photon
        number probability distribution.
        :param port: the output port number or name to compute the statistics for.
        :param parameters: optional parameters to modify the system default parameters.
        :param truncation: the number of probabilities (starting from p(0)) assumed to be non-negligible.
        :return: a probability distribution
        """
        self._make_processor()
        return self._quality_processor.photon_statistics(port, parameters, truncation)

    def beta(self, port: Union[int, str] = None, parameters: dict = None):
        """
        Computes the brightness (the probability of emitting at least one photon) for a port.
        :param port: the port to compute the statistics for.
        :param parameters: optional parameters to modify the system default parameters.
        :return: the brightness
        """
        self._make_processor()
        return self._quality_processor.beta(port, parameters)

    def mu(self, port: Union[int, str] = None, parameters: dict = None, pseudo_limit: float = None):
        """
        Computes the average photon number of light from a port.
        :param port: the source port or list of ports to compute the statistics for.
        :param parameters: optional parameters to modify the system default parameters.
        :param pseudo_limit: the numerical value approximating the zero-efficiency limit.
        :return: the average photon number for the specified port.
        """
        self._make_processor()
        return self._quality_processor.mu(port, parameters, pseudo_limit)

    def g2(self, port: Union[int, str] = None, parameters: dict = None, pseudo_limit: float = None,
           update_mu: bool = False):
        """
        a method simulating a Hanbury-Brown and Twiss setup to compute the integrated intensity correlation: g(2).
        :param port: the source port or list of ports to compute the statistics for.
        :param parameters: optional parameters to modify the system default parameters.
        :param pseudo_limit: the numerical value approximating the zero-efficiency limit.
        :param update_mu: updating quality dictionary with the average photon number used to compute g(2).
        :return: the average photon number or a list of average photon numbers for each port specified.
        """
        self._make_processor()
        return self._quality_processor.g2(port, parameters, pseudo_limit, update_mu)

    def hom(self,
            port: Union[int, str] = None,
            phase: float = None,
            parameters: dict = None,
            pseudo_limit: float = None,
            update_mu: bool = False,
            update_g2: bool = False,
            update_M: bool = True,
            update_coh: bool = True):
        """
        a method simulating a Hong-Ou-Mandel setup to compute the Hong-Ou-Mandel visibility.
           :param port: the source port or list of ports to compute the average photon number for.
           :param phase: the phase of the HOM interferometer giving V_HOM.
           :param parameters: optional parameters to modify the system default parameters.
           :param pseudo_limit: the numerical value approximating the zero-efficiency limit.
           :param update_mu: updating source quality with the average photon number used to compute g(2).
           :param update_g2: updating source quality with the value of g(2) used to compute VHOM.
           :param update_M: updating source quality with the mean wavepacket overlap M.
           :param update_coh: updating source quality with the value of c(1) and c(2) given by HOM simulation.
           :return: the Hong-Ou-Mandel visibility of photons in each port specified.
        """
        self._make_processor()
        return self._quality_processor.hom(port, phase, parameters, pseudo_limit,
                                           update_mu, update_g2, update_M, update_coh)

    def display_hom(self, port: Union[int, str] = None, pseudo_limit=None, parameters: dict = None):
        self._make_processor()
        self._quality_processor.display_hom(port, pseudo_limit, parameters)

    def display_quality(self, port: Union[int, str] = None, pseudo_limit = None, parameters: dict = None):
        self._make_processor()
        self._quality_processor.display_quality(port, pseudo_limit, parameters)

    def lifetime(self,
                 port: Union[int, str] = None,
                 parameters: dict = None,
                 resolution: int = 600,
                 start: float = None,
                 end: float = None,
                 options: Options = None) -> Lifetime:
        self._make_processor()
        return self._quality_processor.lifetime(port, parameters, resolution, start, end, options)

    def plot_lifetime(self,
                      port: Union[int, str, list] = None,
                      parameters: dict = None,
                      resolution: int = 600,
                      start: float = None,
                      end: float = None,
                      label: str = None,
                      scale: float = 1,
                      options: Options = None):
        self._make_processor()
        return self._quality_processor.plot_lifetime(port, parameters, resolution, start, end, label, scale, options)

    def wigner(self, port: Union[int, str] = None, alpha: Union[complex, List[complex]] = 0, parameters: dict = None,
               pseudo_limit: float = 0.01, options: Options = None, lo_resolution=600, lo_fluctuations=2):
        """
        :param port: the port of the source being analysed
        :param alpha: a complex amplitude in phase space or a list of such amplitudes
        :param parameters: a dictionary of parameters to modify the default parameters
        :param pseudo_limit: a loss regime parameter for the pseudo-Wigner algorithm
        :param options: QuTiP options
        :param lo_resolution: the number of numerical points to interpolate the local oscillator shape
        :param lo_fluctuations: the maximum nonlinear fluctuation of local oscillator photons
        :return: the value W(alpha) of the Wigner function at the point alpha
        """
        self._make_processor()
        return self._quality_processor.wigner(port, alpha, parameters, pseudo_limit, options,
                                              lo_resolution, lo_fluctuations)


class GatedSourceComponent(SourceComponent):
    """ A source component with a gate and efficiency"""
    def __init__(self, emitter: EmitterBase, gate: Union[TimeInterval, list, callable] = None,
                 efficiency: float = 1, parameters: dict = None, name: str = None):

        super().__init__(emitter=emitter, parameters=parameters, name=name)
        self.add(0, Gate(interval=gate, modes=emitter.modes))
        self.add(0, UniformLoss(efficiency=efficiency, modes=emitter.modes))
        for port in self.input.ports:
            port.close()
