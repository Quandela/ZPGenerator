from numpy import exp, sqrt, pi


def delay(parameters: dict):
    return [parameters['delay']]


def area(parameters: dict):
    return parameters['area'] * exp(1.j * parameters['phase'])


def square_amplitude(parameters: dict):
    return parameters['area'] / parameters['width'] * exp(1.j * parameters['phase'])


def square_amplitude_detuned(t: float, parameters: dict):
    return parameters['area'] / parameters['width'] * exp(1.j * t * parameters['detuning'] + 1.j * parameters['phase'])


def gaussian(t: float, parameters: dict):
    var = 2 * (parameters['width'] ** 2)
    return parameters['area'] * exp(-(t - parameters['delay']) ** 2 / var +
                                    1.j * t * parameters['detuning'] + 1.j * parameters['phase']) / sqrt(pi * var)


def window(parameters: dict):
    delay = parameters['delay']
    width = parameters['width']
    return [delay - width / 2, delay + width / 2]


def sigma_window(parameters: dict):
    delay = parameters['delay']
    width = parameters['width']
    window = parameters['window']
    return [delay - window * width, delay + window * width]


def amplitude(parameters: dict):
    return parameters['amplitude'] * exp(1.j * parameters['phase'])


def amplitude_detuned(t: float, parameters: dict):
    return parameters['amplitude'] * exp(1.j * t * parameters['detuning'] + 1.j * parameters['phase'])
