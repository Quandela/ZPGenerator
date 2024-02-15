from zpgenerator.system.control import *
from zpgenerator.time import TimeInstantFunction, TimeOperator, TimeIntervalFunction, TimeInterval, TimeFunction, \
    unitary_propagation_superoperator
from qutip import destroy, create, qzero, fock, qeye, sprepost, liouvillian, Qobj
from numpy import pi, exp, sqrt
from math import isclose
from zpgenerator.time.parameters import Parameters


d = Parameters.DELIMITER


def test_channel_base():
    flip = TimeOperator(operator=destroy(2) + create(2),
                        functions=TimeInstantFunction(value=1,
                                                      instant=lambda args: [args['time']],
                                                      parameters={'time': 0},
                                                      name='flip'))
    chn = ChannelBase(operators=flip)
    assert chn.parameters == ['flip' + d + 'time']
    assert chn.default_parameters == {'flip' + d + 'time': 0}
    assert chn.parameter_tree() == {'_TimeOperator': {'flip': {'_default': {'time': 0},
                                                               '_value': {'time': 0},
                                                               '_value (1)': {'time': 0}}}}

    assert chn.support(0)
    assert chn.support(1, {'flip' + d + 'time': 1})
    assert not chn.is_dirac(1)
    assert chn.is_dirac(1, {'flip' + d + 'time': 1})

    assert chn.evaluate(0) == qzero(2)
    assert chn.evaluate_dirac(0).evaluate() == unitary_propagation_superoperator(destroy(2) + create(2))
    assert chn.evaluate_dirac(1).evaluate() == sprepost(qeye(2), qeye(2))
    assert chn.evaluate_dirac(1, {'flip' + d + 'time': 1}).evaluate() == \
           unitary_propagation_superoperator(destroy(2) + create(2))


def _make_ham_env_chn():
    def gaussian(t, args):
        return args['area'] * exp(-t ** 2 / (2 * args['width'])) / sqrt(2 * pi * args['width'] ** 2)

    ham = HamiltonianBase()
    window = TimeInterval(lambda args: [args['delay'] - 6 * args['width'], args['delay'] + 6 * args['width']],
                          parameters={'delay': 0, 'width': 1})
    rabi = TimeFunction(gaussian, parameters={'area': pi, 'width': 1})
    func = TimeIntervalFunction(value=rabi, interval=window)
    ham.add(TimeOperator((create(2) + destroy(2)) / 2, functions=func))

    env = EnvironmentBase()
    rabi_sq = TimeFunction(lambda t, args: args['phonon_coefficient'] * gaussian(t, args) ** 2,
                           parameters={'area': pi, 'width': 1, 'phonon_coefficient': 10e-4})
    gaussian_sq = TimeIntervalFunction(value=rabi_sq, interval=window)
    env.add(TimeOperator(create(2) * destroy(2), functions=gaussian_sq))

    chn = ChannelBase()
    chn.add(TimeOperator.dirac(lambda args: (destroy(2) + create(2)), time='time', parameters={'time': 1}), name='flip')
    return ham, env, chn


def test_control_base():
    ham, env, chn = _make_ham_env_chn()
    ctrl = ControlBase(hamiltonian=ham, environment=env, channel=chn, name='drive')
    rabi0 = pi / sqrt(2 * pi)
    assert ctrl.hamiltonian.evaluate(0) == liouvillian(rabi0 * (create(2) + destroy(2)) / 2)
    assert ctrl.environment.evaluate(0) == liouvillian(qzero(2), [rabi0 ** 2 * 10e-4 * create(2) * destroy(2)])
    assert ctrl.channel.evaluate(0) == qzero(2)
    assert ctrl.channel.evaluate_dirac(1).evaluate() == unitary_propagation_superoperator(destroy(2) + create(2))
    assert ctrl.evaluate(0) == liouvillian(H=rabi0 * (create(2) + destroy(2)) / 2,
                                           c_ops=[rabi0 ** 2 * 10e-4 * create(2) * destroy(2)])
    assert ctrl.evaluate_dirac(1).evaluate() == unitary_propagation_superoperator(destroy(2) + create(2))
    assert ctrl.is_time_dependent(0)
    assert ctrl.parameters == ['area',
                               'delay',
                               'flip' + d + 'time',
                               'phonon_coefficient',
                               'width']
    assert ctrl.parameter_tree() == {'_channel': {'flip': {'_TimeInstantFunction': {'_value (1)': {'time': 1}},
                                                           '_operator': {'time': 1}}},
                                     '_environment': {
                                         '_TimeOperator': {'_TimeIntervalFunction': {'_interval': {'delay': 0,
                                                                                                   'width': 1},
                                                                                     '_value': {
                                                                                         'area': pi,
                                                                                         'phonon_coefficient': 10e-4,
                                                                                         'width': 1}}}},
                                     '_hamiltonian': {
                                         '_TimeOperator': {'_TimeIntervalFunction': {'_interval': {'delay': 0,
                                                                                                   'width': 1},
                                                                                     '_value': {
                                                                                         'area': pi,
                                                                                         'width': 1}}}}}


def test_controlled_system_init_empty():
    sys = ControlledSystem()
    assert sys.dim is None
    assert sys.subdims is None
    assert sys.parameters == []
    assert sys.evaluate(0) == Qobj()


def _make_controlled_system():
    ctrl = ControlBase(*_make_ham_env_chn(), name='pulse')

    ham = HamiltonianBase()
    ham.add(lambda args: args['detuning'] * create(2) * destroy(2), parameters={'detuning': 0})

    env = EnvironmentBase()
    env.add(lambda args: args['rate'] * destroy(2), parameters={'rate': 1}, name='decay')
    env.add(lambda args: args['rate'] * create(2) * destroy(2), parameters={'rate': 0}, name='dephasing')

    return ControlledSystem(hamiltonian=ham, environment=env, control=ctrl)


def test_controlled_system():
    sys = _make_controlled_system()

    sys.states.update({'|g>': fock(2, 0), '|e>': fock(2, 1)})
    sys.operators.update({'dipole': destroy(2), 'X': create(2) + destroy(2), 'num': create(2) * destroy(2)})
    sys.name = 'tls'

    assert sys.dim == 2
    assert sys.subdims == [2]
    assert not sys.is_super
    assert sys.name == 'tls'
    assert sys.parameters == ['decay' + d + 'rate',
                              'dephasing' + d + 'rate',
                              'detuning',
                              'pulse' + d + 'area',
                              'pulse' + d + 'delay',
                              'pulse' + d + 'flip' + d + 'time',
                              'pulse' + d + 'phonon_coefficient',
                              'pulse' + d + 'width']
    assert sys.parameter_tree() == {
        '_control': {'pulse': {'_channel': {'flip': {'_TimeInstantFunction': {'_value (1)': {'time': 1}},
                                                     '_operator': {'time': 1}}},
                               '_environment': {'_TimeOperator': {'_TimeIntervalFunction': {'_interval': {'delay': 0,
                                                                                                          'width': 1},
                                                                                            '_value': {'area': pi,
                                                                                                       'phonon_coefficient': 0.001,
                                                                                                       'width': 1}}}},
                               '_hamiltonian': {'_TimeOperator': {'_TimeIntervalFunction': {'_interval': {'delay': 0,
                                                                                                          'width': 1},
                                                                                            '_value': {'area': pi,
                                                                                                       'width': 1}}}}}},
        '_environment': {'_TimeOperator': {'decay': {'rate': 1}},
                         '_TimeOperator (1)': {'dephasing': {'rate': 0}}},
        '_hamiltonian': {'_TimeOperator': {'_operator': {'detuning': 0}}}}
    assert sys.has_interval
    assert sys.has_instant
    assert sys.times() == [-6, 1, 6]
    assert sys.support(0)
    assert sys.is_time_dependent(-6)
    assert sys.is_time_dependent(0)
    assert not sys.is_time_dependent(6)

    # recognized as nonherm time dependent even if it vanishes... fine for now but should be fixed to optimize
    assert sys.is_nonhermitian_time_dependent(0, {'phonon_coefficient': 0})
    assert not sys.is_dirac(0)
    assert sys.is_dirac(1)

    assert sys.states['|g>'] == fock(2, 0)
    assert sys.states['|e>'] == fock(2, 1)
    assert sys.operators['X'] == destroy(2) + create(2)

    assert sys.evaluate(0, {'detuning': 2, 'dephasing' + d + 'rate': 1}) == \
           liouvillian(H=2 * sys.operators['num'] + sqrt(pi / 8) * sys.operators['X'],
                       c_ops=[sys.operators['dipole'], sys.operators['num'], 10e-4 * pi / 2 * sys.operators['num']])
    assert sys.evaluate_quadruple(0).hamiltonian.list_form()[1][0] == sys.operators['X'] / 2
    assert isclose(sys.evaluate_quadruple(0).hamiltonian.list_form()[1][1](0, {}), sqrt(pi / 2))
    assert sys.evaluate_dirac(1).evaluate() == unitary_propagation_superoperator(sys.operators['X'])
