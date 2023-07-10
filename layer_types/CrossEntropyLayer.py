from layer_types.Layer import Layer
from data_structure.Tensor import Tensor
from data_structure.Shape import Shape
import numpy as np

class CrossEntropyLayer(Layer):

    def __init__(self, inTensor):
        self.inTensor = inTensor

    def forward(self, inTensor, label):
        self.inTensor = inTensor
        self.label = label

        elements = self.inTensor.get_elements()
        loss = 0
        for i in list(range(self.inTensor.get_shape().get_axis()[1])):
            loss = loss + label[i] * np.log(elements[i])
        loss = loss * -1
        return loss

    def backward(self, inTensor, label):
        self.inTensor = inTensor
        elements = self.inTensor.get_elements()
        result = [0] * len(label)
        for i in range(len(label)):
            result[i] = -label[i] / elements[i]
        self.inTensor.set_deltas(result)
        return inTensor
