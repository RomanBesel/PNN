from data_structure.Shape import Shape

class Tensor():

    elements = []
    shape = None
    deltas = []

    def __init__(self, elements, shape, deltas):
        self.elements = elements
        self.shape = shape
        self.deltas = deltas

    def get_elements(self):
        return self.elements

    def set_elements(self, elements):
        self.elements = elements

    def get_shape(self):
        return self.shape

    def set_shape(self, shape):
        self.shape = shape

    def get_deltas(self):
        return self.deltas

    def set_deltas(self, deltas):
        self.deltas = deltas
