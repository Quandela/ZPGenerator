from ...network import Component
from ...misc.display import Display


class CircuitComponent(Component):

    def display(self):
        Display(self).display()
