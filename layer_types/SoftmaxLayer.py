from layer_types.Layer import Layer
from data_structure.Tensor import Tensor
from data_structure.Shape import Shape
import numpy as np

class SoftmaxLayer(Layer):

    def __init__(self, inTensor, outTensor):
        self.inTensor = inTensor
        self.outTensor = outTensor

    def forward(self, inTensor, outTensor):
        self.inTensor = inTensor
        value = self.inTensor.get_elements()
        e_x = np.exp(value)
        result = e_x / e_x.sum(axis = 0)
        outTensor.set_elements(result)
        return outTensor

    def backward(self, outTensor, inTensor):
        elements_out = outTensor.get_elements()
        size1 = len(outTensor.get_elements())
        size2 = len(inTensor.get_elements())
        helper_matrix = np.zeros([size1, size2])

        for i in range(size1):
            for j in range(size2):
                if i == j:
                        kroneckerdelta = 1
                else: kroneckerdelta = 0

                helper_matrix[i, j] = elements_out[i] * (kroneckerdelta - elements_out[j])
        inTensor.set_deltas(np.dot(outTensor.get_deltas(), helper_matrix))
        return inTensor


