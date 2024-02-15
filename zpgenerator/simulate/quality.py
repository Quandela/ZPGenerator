from .algorithms import compute_photon_number_distribution, compute_brightness, estimate_average_photon_number, \
    estimate_intensity_correlation, estimate_hom_visibility, estimate_hom_visibility_with_coherence, compute_lifetime, \
    compute_wigner_function
from typing import List
from qutip import Options
from .base_processor import ProcessorBase
from ..network import AComponent
from ..system import AElement
from ..time import Lifetime
from typing import Union
from numpy import real


class ProcessorQuality(ProcessorBase):
    """Adds useful methods to determine figures of merit"""

    def __init__(self, component: Union[AElement, AComponent] = None):
        super().__init__(component)
        self.quality = {}

    def _estimate_truncation(self, port: int, parameters: dict = None):
        truncation = 2 * (1 + max(1, round(self.mu(port, parameters))))
        return max(truncation, 2)

    def _estimate_pseudo_limit(self, port: int, parameters: dict = None):
        return 0.005  # This should be more sophisticated

    def _name_to_port(self, port):
        port = 0 if port is None else self.component.get_port_number(port)
        name = str(port)
        assert self.component.output.ports[self.component.unmasked_position(port)].is_open, "Can only compute the quality of open ports"
        return name, port

    def _update_quality(self, quality: dict, name: str):
        if name in self.quality.keys():
            self.quality[name].update(quality)
        else:
            self.quality.update({name: quality})

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
        name, port = self._name_to_port(port)

        truncation = self._estimate_truncation(port, parameters) if truncation is None else truncation

        distribution = compute_photon_number_distribution(source=self, port=port, truncation=truncation,
                                                          parameters=parameters)
        # distribution.precision_check(self.precision)

        labels = ['pn', 'beta', 'mu', 'g2']
        mu = distribution.mu()
        quality = {labels[0]: distribution,
                   labels[1]: distribution.beta(),
                   labels[2]: mu}

        if mu > 10 ** -self.precision:
            quality.update({labels[3]: distribution.g2()})
        else:
            print("Warning: no light detected in mode " + ('' if self.modes == 1 else str(port)) + ', ' +
                  ('g2' if self.modes == 1 else 'g2 ' + str(port)) +
                  " cannot be defined.")

        self._update_quality(quality, name)

        return distribution

    def beta(self, port: Union[int, str] = None, parameters: dict = None):
        """
        Computes the brightness (the probability of emitting at least one photon) for a port.
        :param port: the port to compute the statistics for.
        :param parameters: optional parameters to modify the system default parameters.
        :return: the brightness
        """
        name, port = self._name_to_port(port)
        beta = compute_brightness(self.component, port, parameters)
        self._update_quality({'beta': beta}, name)
        return beta

    def mu(self, port: Union[int, str] = None, parameters: dict = None, pseudo_limit: float = None):
        """
        Computes the average photon number of light from a port.
        :param port: the source port or list of ports to compute the statistics for.
        :param parameters: optional parameters to modify the system default parameters.
        :param pseudo_limit: the numerical value approximating the zero-efficiency limit.
        :return: the average photon number for the specified port.
        """
        name, port = self._name_to_port(port)
        pseudo_limit = self._estimate_pseudo_limit(port, parameters) if pseudo_limit is None else pseudo_limit
        mu = estimate_average_photon_number(self.component, port=port, pseudo_limit=pseudo_limit,
                                            parameters=parameters)
        self._update_quality({'mu': mu}, name)
        return mu

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
        name, port = self._name_to_port(port)
        pseudo_limit = self._estimate_pseudo_limit(port, parameters) if pseudo_limit is None else pseudo_limit

        # Simulate using a detector with threshold resolution in the lossy regime
        g2, mu = estimate_intensity_correlation(self.component, port=port, pseudo_limit=pseudo_limit,
                                                parameters=parameters)
        if update_mu:
            self._update_quality({'mu': mu}, name)

        self._update_quality({'g2': g2}, name)
        return g2

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
        name, port = self._name_to_port(port)

        assert not (update_mu is True and update_g2 is False), "Cannot update mu without updating g2, use mu() instead."
        assert all(u is False for u in [update_mu, update_g2, update_M, update_coh]) if phase is not None else True, \
            "Cannot update other figures of merit when computing VHOM for a specified phase."

        pseudo_limit = self._estimate_pseudo_limit(port, parameters) if pseudo_limit is None else pseudo_limit

        labels = ['vhom', 'M', 'c1', 'c2', 'mu', 'g2']

        quality = {}

        if phase is None:
            if update_coh:
                vhom, c1, c2 = estimate_hom_visibility_with_coherence(self.component, port, pseudo_limit, parameters)
                quality[labels[2]] = real(c1)
                quality[labels[3]] = real(c2)
            else:
                vhom = estimate_hom_visibility(self.component, port, pseudo_limit, parameters)

            if update_g2 or labels[-1] not in self.quality.keys():
                self.g2(port=port, parameters=parameters, pseudo_limit=pseudo_limit,
                        update_mu=update_mu if labels[-2] in self.quality.keys() else True)

            quality[labels[0]] = real(vhom)
            if update_M:
                quality[labels[1]] = real(vhom + self.quality[name]['g2'])

            self._update_quality(quality, name)

        else:  # estimate VHOM(phi)
            vhom = estimate_hom_visibility(self.component, port, pseudo_limit, parameters, phase)
            quality[labels[0]] = real(vhom)

            self._update_quality(quality, name)

        return quality

    def display_hom(self, port: Union[int, str] = None, pseudo_limit = None, parameters: dict = None):
        quantities = self.hom(port, parameters=parameters, pseudo_limit=pseudo_limit)
        print("{:<30} | {:}".format("Figure of Merit", "Value"))
        labels = {'M': 'Mean wavepacket overlap',
                  'vhom': 'Hong-Ou-Mandel visibility',
                  'c1': 'First order number coherence',
                  'c2': 'Second order number coherence'}
        for k, v in quantities.items():
            print("{key:<30} | {number:.{digits}f}".format(key=labels[k], number=v, digits=self.precision - 2))
        print()

    def display_quality(self, port: Union[int, str] = None, pseudo_limit = None, parameters: dict = None):
        pn = self.photon_statistics(port, parameters=parameters)
        pn.display()
        pn.display_figures()
        self.display_hom(port, pseudo_limit=pseudo_limit, parameters=parameters)

    def lifetime(self,
                 port: Union[int, str] = None,
                 parameters: dict = None,
                 resolution: int = 600,
                 start: float = None,
                 end: float = None,
                 options: Options = None) -> Lifetime:
        name, port = self._name_to_port(port)
        lifetime = compute_lifetime(self.component, port, resolution, start, end, parameters, options)
        self.quality.update({name: {'lifetime': lifetime}})

        return lifetime

    def plot_lifetime(self,
                      port: Union[int, str, list] = None,
                      parameters: dict = None,
                      resolution: int = 600,
                      start: float = None,
                      end: float = None,
                      label: str = None,
                      scale: float = 1,
                      options: Options = None):
        if port == 'All' or isinstance(port, list):
            ports = list(range(self.modes)) if port == 'All' else port
            return [self.plot_lifetime(i, parameters=parameters, resolution=resolution,
                                       start=start, end=end, options=options) for i in ports]
        else:
            name, port = self._name_to_port(port)

            if all(arg is None for arg in [port, resolution, start, end, options]):
                lifetime = self.quality[name]['lifetime']
            else:
                lifetime = self.lifetime(port=port, resolution=resolution, start=start, end=end,
                                         parameters=parameters, options=options)

            return lifetime.plot(label if label else name, scale=scale)

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
        name, port = self._name_to_port(port)

        # get the LO profile from the source lifetime
        lifetime = self.lifetime(port=port, resolution=lo_resolution, parameters=parameters, options=options)
        wigner = compute_wigner_function(self.component, port, lifetime, alpha, parameters,
                                         pseudo_limit, lo_resolution, lo_fluctuations)
        self.quality.update({name: {'wigner': wigner}})
        return wigner


