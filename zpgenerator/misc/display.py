from ..network.element import AElement
from ..network.component import Component
from ..time import Parameters


dp = Parameters.DEFAULT_PREFIX

class Display:
    """Displays elements, components, and networks"""
    def __init__(self, component: AElement, ports: list = None):
        self.component_name = component.default_name[1:] if component.name is None else component.name
        self.component = component if isinstance(component, Component) else Component(component)
        self.ports = ['(' + str(p) + ')' for p in ports] if ports else ['---'] * self.component.modes

    def display(self):
        name_len = len(self.component_name)
        print('')
        print('        ' + '_' * (name_len + 4))
        for i in range(self.component.modes):
            input_port = self.component.input.ports[i]
            output_port = self.component.output.ports[i]
            input_port_str = '|0>--' if input_port.is_closed else '   --'

            output_port_str = '--'
            if output_port.is_closed and not output_port.is_monitored:
                output_port_str += '|'
            elif output_port.is_monitored:
                output_port_str = output_port_str + 'D~'
                bin_names = []
                for name in output_port.bin_names:
                    if name[:len(dp)] != dp:
                        bin_names.append(name)
                if bin_names:
                    output_port_str += ' [' + ', '.join(bin_names) + ']'

            name = ' ' if i != round((self.component.modes - 1)/2) else self.component_name
            comp_str = self.ports[i][:-1] + '|' + name.center(name_len + 4, ' ') + '|' + self.ports[i][1:]

            print(input_port_str + comp_str + output_port_str)
        print('        ' + chr(8254) * (name_len + 4))


def display(component: AElement):
    Display(component).display()
