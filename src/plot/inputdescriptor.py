from plot.operator import Getter


class InputDescriptor:
    def __init__(self, label, getter: Getter):
        self.label = label
        self.getter = getter

    @property
    def mapper(self):
        return self.getter.mapper


def create_descriptor_set(**descriptors:InputDescriptor):
    return descriptors
