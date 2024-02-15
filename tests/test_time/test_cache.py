from zpgenerator import *
from qutip import Qobj


def test_default_cache():
    source = Source.two_level(pulse=Pulse.gaussian())
    source.mu()
