from ...network import Component
from ...misc.display import Display


class DetectorComponent(Component):
    """ A component with open inputs and outputs that are either closed or monitored """

    def display(self):
        Display(self).display()