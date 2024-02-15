from numpy import pi, exp, cosh, sinh, linspace, sin, cos, sqrt, real, imag
from scipy.interpolate import interp1d
from scipy.integrate import simps
from qutip import Qobj, spre, spost, sprepost
from ...time import PulseBase, TimeOperator
from ...system import EnvironmentBase
from functools import lru_cache, cached_property


class Material:
    BOLTZMANN_CONSTANT = 1.380649 * 10 ** (-23)  # J/K
    PLANCK_CONSTANT = 1.054571817 * 10 ** (-34)  # Js
    ELEMENTARY_CHARGE = 1.60217663 * 10 ** (-19)  # C

    def __init__(self,
                 density,
                 speed_of_sound,
                 electron_deformation_constant,
                 hole_deformation_constant,
                 electron_confinement,
                 hole_confinement,
                 timescale=10 ** -12):
        self.density = density
        self.speed_of_sound = speed_of_sound
        self.electron_deformation_constant = electron_deformation_constant
        self.hole_deformation_constant = hole_deformation_constant
        self.electron_confinement = electron_confinement
        self.hole_confinement = hole_confinement
        self.timescale = timescale
        self.alpha = 1 / (4 * (pi ** 2) * self.density * self.PLANCK_CONSTANT * (self.speed_of_sound ** 5))

    def spectral_density(self, omega: float):
        omega = omega / self.timescale
        jspec = self.alpha * omega ** 3 * \
                (self.electron_deformation_constant *
                 exp(-omega ** 2 * self.electron_confinement ** 2 / (4 * self.speed_of_sound ** 2))
                 - self.hole_deformation_constant *
                 exp(-omega ** 2 * self.hole_confinement ** 2 / (4 * self.speed_of_sound ** 2))) ** 2
        return self.timescale * jspec

    @cached_property
    def polaron_shift(self):
        sos = self.speed_of_sound ** 2
        return (sqrt(pi) / 2) * self.alpha * (
                sqrt(2) * self.electron_deformation_constant ** 2 / (self.electron_confinement ** 2 / sos) ** (3 / 2) +
                sqrt(2) * self.hole_deformation_constant ** 2 / (self.hole_confinement ** 2 / sos) ** (3 / 2) -
                8 * self.electron_deformation_constant * self.hole_deformation_constant /
                ((self.electron_confinement ** 2 + self.hole_confinement ** 2) / sos) ** (3 / 2)) * self.timescale

    @classmethod
    def ingaas_quantum_dot(cls, electron_confinement=4.2e-9, hole_confinement=4.2e-9):
        return cls(density=5370,
                   speed_of_sound=5110,
                   electron_deformation_constant=7 * cls.ELEMENTARY_CHARGE,
                   hole_deformation_constant=-3.5 * cls.ELEMENTARY_CHARGE,
                   electron_confinement=electron_confinement,
                   hole_confinement=hole_confinement)


class PhononBath:
    """
    A collection of parameters and methods to evaluate the environmental impact of a phonon bath on an emitter
    """

    def __init__(self,
                 material: Material,
                 temperature: float,
                 resolution: int = 150,
                 max_power: float = 15):
        self.material = material
        self._temperature = temperature
        self.resolution = resolution
        self.max_power = max_power

        self._temp_con = self.material.PLANCK_CONSTANT / \
                         (2 * self.material.BOLTZMANN_CONSTANT * self.temperature * self.material.timescale)

        self._rs_func = None
        self._ic_func = None
        self._phi0 = None
        self._attenuation = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature
        self._rs_func = None
        self._ic_func = None

    @property
    def polaron_shift(self):
        return self.material.polaron_shift

    @property
    def attenuation(self):
        return self._attenuation

    def _rs(self, rabi_r: float):
        if not self._rs_func:
            self.initialize()
        return real(self._rs_func(rabi_r))

    def _is(self, rabi_r: float):
        return self._is_func(rabi_r)

    @lru_cache(maxsize=1)
    def _rc(self, rabi_r: float):
        return self._rc_func(rabi_r)

    def _ic(self, rabi_r: float):
        if not self._ic_func:
            self.initialize()
        return real(self._ic_func(rabi_r))

    @staticmethod
    def coth(x):
        return cosh(x) / sinh(x)

    def _rs_integrand(self, delay, freq, rabi_r):
        return self.material.spectral_density(freq) * \
            self.coth(self._temp_con * freq) * cos(freq * delay) * sin(rabi_r * delay) / 2 if rabi_r else \
            0 * sin(delay) * sin(freq)

    def _is_func(self, rabi_r: float):
        return - (pi / 4) * self.material.spectral_density(rabi_r)

    def _rc_func(self, rabi_r: float):
        return (pi / 4) * self.material.spectral_density(rabi_r) * self.coth(self._temp_con * rabi_r) if rabi_r else 0

    def _ic_integrand(self, delay, freq, rabi_r):
        return -self.material.spectral_density(freq) * sin(freq * delay) * cos(rabi_r * delay) / 2

    def _phi0_integrand(self, freq):
        return self.coth(self._temp_con * freq) * self.material.spectral_density(freq) / freq ** 2 if freq else 0

    def initialize(self):

        power_space = linspace(0, self.max_power, self.resolution)
        delay_space = linspace(0, 10, self.resolution)
        freq_space = linspace(1e-16, 10, self.resolution)

        rs_set = []
        for rabi_r in power_space:
            rs_eval = self._rs_integrand(delay_space[:, None], freq_space, rabi_r)
            rs_set.append(simps(simps(rs_eval, freq_space), delay_space))
        self._rs_func = interp1d(power_space, rs_set)

        ic_set = []
        for rabi_r in power_space:
            ic_eval = self._ic_integrand(delay_space[:, None], freq_space, rabi_r)
            ic_set.append(simps(simps(ic_eval, freq_space), delay_space))
        self._ic_func = interp1d(power_space, ic_set)

        self._phi0 = simps([self._phi0_integrand(f) for f in freq_space], freq_space)
        self._attenuation = exp(-self._phi0 / 2)

    def rabi_r(self, rabi_x: float, rabi_y: float, detuning: float):
        return sqrt(rabi_x ** 2 + rabi_y ** 2 + detuning ** 2)

    @lru_cache(maxsize=1)
    def rabi_set(self, rabi, detuning: float):
        rabi_x = real(rabi)
        rabi_y = imag(rabi)
        rabi_r = self.rabi_r(rabi_x, rabi_y, detuning)
        return rabi_x, rabi_y, rabi_r

    @lru_cache(maxsize=1)
    def c1(self, rabi_r: float):
        return self._rs(rabi_r) / rabi_r

    @lru_cache(maxsize=1)
    def c2(self, rabi_r: float, detuning: float):
        return detuning * self._rc(rabi_r) / rabi_r ** 2

    @lru_cache(maxsize=1)
    def c3(self, rabi_r: float):
        return self._is(rabi_r) / rabi_r

    @lru_cache(maxsize=1)
    def c4(self, rabi_r: float, detuning: float):
        return detuning * (self.polaron_shift / 2 + self._ic(rabi_r)) / rabi_r ** 2

    def cx_plus(self, rabi, parameters: dict = None):
        detuning = self.polaron_shift + parameters['resonance']
        rabi_x, rabi_y, rabi_r = self.rabi_set(rabi, detuning)
        return rabi_y * self.c1(rabi_r) - rabi_x * self.c2(rabi_r, detuning) if rabi_r else 0

    def cx_minus(self, rabi, parameters: dict = None):
        detuning = self.polaron_shift + parameters['resonance']
        rabi_x, rabi_y, rabi_r = self.rabi_set(rabi, detuning)
        return 1.j * (rabi_y * self.c3(rabi_r) - rabi_x * self.c4(rabi_r, detuning)) if rabi_r else 0

    def cy_plus(self, rabi, parameters: dict = None):
        detuning = self.polaron_shift + parameters['resonance']
        rabi_x, rabi_y, rabi_r = self.rabi_set(rabi, detuning)
        return -(rabi_x * self.c1(rabi_r) + rabi_y * self.c2(rabi_r, detuning)) if rabi_r else 0

    def cy_minus(self, rabi, parameters: dict = None):
        detuning = self.polaron_shift + parameters['resonance']
        rabi_x, rabi_y, rabi_r = self.rabi_set(rabi, detuning)
        return -1.j * (rabi_x * self.c3(rabi_r) + rabi_y * self.c4(rabi_r, detuning)) if rabi_r else 0

    def _gamma_star_over_rc(self, rabi_x: float, rabi_y: float, rabi_r: float):
        return 4 * (rabi_x ** 2 + rabi_y ** 2) / rabi_r ** 2 if rabi_r else 0

    def gamma_star(self, rabi, parameters: dict = None):
        detuning = self.polaron_shift + parameters['resonance']
        rabi_x, rabi_y, rabi_r = self.rabi_set(rabi, detuning)
        return sqrt(self._rc(rabi_r) * self._gamma_star_over_rc(rabi_x, rabi_y, rabi_r))

    def op_plus(self, pauli: Qobj, num: Qobj):
        return sprepost(pauli, num) - spre(num * pauli) + sprepost(num, pauli) - spost(pauli * num)

    def op_minus(self, pauli: Qobj, num: Qobj):
        return sprepost(pauli, num) - spre(num * pauli) - sprepost(num, pauli) + spost(pauli * num)

    def build_environment(self, pulse: PulseBase, transition: Qobj):
        op_x = transition + transition.dag()
        op_y = 1.j * transition - 1.j * transition.dag()
        num = transition.dag() * transition

        parameters = {'resonance': 0}

        cache = True
        pulse.cache = True
        cx_plus = PulseBase(pulse, cache=cache)
        cx_plus.compose_with(self.cx_plus, parameters)

        cx_minus = PulseBase(pulse, cache=cache)
        cx_minus.compose_with(self.cx_minus, parameters)

        cy_plus = PulseBase(pulse, cache=cache)
        cy_plus.compose_with(self.cy_plus, parameters)

        cy_minus = PulseBase(pulse, cache=cache)
        cy_minus.compose_with(self.cy_minus, parameters)

        gamma_star = PulseBase(pulse, cache=cache)
        gamma_star.compose_with(self.gamma_star, parameters)

        environment = EnvironmentBase()
        environment.add(TimeOperator(operator=self.op_plus(op_x, num), functions=cx_plus))
        environment.add(TimeOperator(operator=self.op_minus(op_x, num), functions=cx_minus))
        environment.add(TimeOperator(operator=self.op_plus(op_y, num), functions=cy_plus))
        environment.add(TimeOperator(operator=self.op_minus(op_y, num), functions=cy_minus))
        environment.add(TimeOperator(operator=num, functions=gamma_star))

        return environment
