from ...time import merge_times, Lifetime
from ...virtual import VState, Generator
from ...network import AComponent
from qutip import Options
from numpy import linspace


def compute_lifetime(source: AComponent,
                     port: int = None,
                     resolution: int = 600,
                     start: float = None,
                     end: float = None,
                     parameters: dict = None,
                     options: Options = None):
    options = Options(nsteps=50000) if options is None else options  # set options for qutip
    options.store_states = True

    times = source.times(parameters)  # determine simulation stop times
    initial_time = source.initial_time if source.initial_time else times[0] if times else 0
    start = start if start is not None else initial_time
    end = end if end else times[-1] if times else 10
    endpoints = [start, end]
    times = merge_times([times, endpoints])
    diff = end - start

    # initialize the state
    state = VState(state=source.initial_state, time=min(initial_time, start))

    # initialize arrays for expectations
    eoptimes = []
    eopvalues = []

    generator = Generator(component=source, lifetime_mode=port)

    # Main propagation algorithm
    for i in range(1, len(times)):  # Propagate from initial time to final time
        t0 = times[i - 1]  # current time
        t1 = times[i]  # next stop time

        if source.is_dirac(t0, parameters):  # if we have instant operators
            dirac_operator = source.evaluate_dirac(t0, parameters).evaluate()
            state.apply_operator(dirac_operator)

        # build propagator
        propagator = generator.build_propagator(t=t0, parameters=parameters, options=options)

        if t0 < endpoints[0]:  # propagate normally
            state.propagate(propagator, t1)
        elif endpoints[0] <= t0 < endpoints[-1]:  # evaluate lifetimes and append results
            tlist = list(linspace(t0, t1, round((t1 - t0) / diff * resolution) + 2))
            result = state.propagate(propagator, t1, tlist=tlist)
            if tlist:
                eoptimes = eoptimes[0:-1] + result.times
                eopvalues = eopvalues[0:-1] + list(result.expect[0])
        else:
            break

    return Lifetime(times=eoptimes, population=eopvalues)
