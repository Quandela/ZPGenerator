from ..components.sources import *
from ..time import TimeInterval, PulseBase, Lifetime, parinit
from ..dynamic.operator.phonon_bath import Material
from typing import Union
from qutip import Qobj


class Source(SourceComponent):
    """
    A source factory
    """

    @classmethod
    def two_level(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                  parameters: dict = None, name: str = None, emitter_name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return TwoLevelSource(pulse=pulse, gate=gate, efficiency=efficiency, parameters=parameters, name=name,
                              emitter_name=emitter_name)

    @classmethod
    def purcell(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                purcell_factor: float = None, regime: float = None, timescale: float = None,
                parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        source = PurcellSource(pulse=pulse, gate=gate, efficiency=efficiency,
                               purcell_factor=purcell_factor, regime=regime, timescale=timescale,
                               parameters=parameters, name=name)
        source.output.ports[0].close()
        source.mask()
        return source

    @classmethod
    def phonon_assisted(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                        purcell_factor: float = 10, regime: float = 0.1, timescale: float = 1,
                        temperature: float = 4, material: Material = Material.ingaas_quantum_dot(),
                        resolution: int = 300, max_power: float = 30,
                        parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        source = PhononAssistedSource(pulse=pulse, gate=gate, efficiency=efficiency,
                                      purcell_factor=purcell_factor, regime=regime, timescale=timescale,
                                      temperature=temperature, material=material,
                                      resolution=resolution, max_power=max_power,
                                      parameters=parameters, name=name)
        source.output.ports[0].close()
        source.mask()
        return source

    @classmethod
    def exciton(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return ExcitonSource(pulse=pulse, gate=gate, efficiency=efficiency,
                             parameters=parameters, name=name)

    @classmethod
    def biexciton(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                  parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return BiexcitonSource(pulse=pulse, gate=gate, efficiency=efficiency,
                               parameters=parameters, name=name)

    @classmethod
    def trion(cls, charge: str = 'negative', pulse: PulseBase = None, pulse_orthogonal: PulseBase = None,
              gate: Union[TimeInterval, list] = None, efficiency: float = 1, parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return TrionSource(charge=charge, pulse=pulse, pulse_orthogonal=pulse_orthogonal,
                           gate=gate, efficiency=efficiency,
                           parameters=parameters, name=name)

    @classmethod
    def fock(cls, state: Union[int, Qobj], gate: Union[TimeInterval, list] = None,
             shape: Union[PulseBase, Lifetime] = None, shape_resolution: int = 1000, efficiency: float = 1,
             parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return FockSource(state=state, gate=gate, shape=shape, shape_resolution=shape_resolution,
                          efficiency=efficiency,
                          parameters=parameters, name=name)

    @classmethod
    def shaped_laser(cls,
                     shape: Union[PulseBase, Lifetime, callable] = None,
                     resolution: int = 1000,
                     truncation: int = 2,
                     gate: Union[TimeInterval, list] = None,
                     efficiency: float = 1,
                     parameters: dict = None,
                     name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return ShapedLaserSource(shape=shape, resolution=resolution, truncation=truncation, gate=gate,
                                 efficiency=efficiency,
                                 parameters=parameters, name=name)

    @classmethod
    def perceval(cls,
                 emission_probability: float = 1,
                 multiphoton_component: float = 0,
                 indistinguishability: float = 1,
                 name: str = None):
        return DistinguishableNoiseSource(emission_probability=emission_probability,
                                          multiphoton_component=multiphoton_component,
                                          indistinguishability=indistinguishability,
                                          name=name)
