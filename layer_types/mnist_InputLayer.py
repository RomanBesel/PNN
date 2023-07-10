from layer_types.InputLayer import InputLayer
from data_structure.Tensor import Tensor
from data_structure.Shape import Shape
import numpy as np


class mnist_InputLayer(InputLayer):

    def __init__(self):
        'Constructor'
    def get_train_data(self, data):
        data[0] = data[0]/255
        return data[0]

    def get_label(self, data):
        return self.labelsfor(data[1])

    def labelsfor(self, i):
        switcher = {
            0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        return switcher.get(i)