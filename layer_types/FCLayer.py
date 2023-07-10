from layer_types.Layer import Layer
from data_structure.Shape import Shape
from data_structure.Tensor import Tensor
import numpy as np

class FCLayer(Layer):

    weights = []
    bias = []

    def __init__(self, inTensor, outTensor):
        self.inTensor = inTensor
        self.outTensor = outTensor
        inShape = inTensor.get_shape().get_axis()[1]
        outShape = outTensor.get_shape().get_axis()[1]
        self.weights = 2 * np.random.rand(inShape, outShape) - 1
        self.bias = 2 * np.random.rand(1, outShape) - 1

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_bias(self):
        return self.bias

    def set_bias(self, bias):
        self.bias = bias

    def forward(self, inTensor, outTensor):
        self.inTensor = inTensor
        self.outTensor = outTensor
        result = (np.dot(self.inTensor.get_elements(), self.weights) + self.bias)
        outTensor.set_elements(result[0])
        return outTensor

    def backward(self, outTensor, inTensor):
        inTensor.set_deltas(np.dot(outTensor.get_deltas(), np.transpose(self.weights)))
        return inTensor

    def calculate_delta_weights(self, inTensor, outTensor, alpha):
        elements_col = np.reshape(inTensor.get_elements(), (-1, 1))
        deltas_row = np.reshape(outTensor.get_deltas(), (1, -1))
        weight_error = np.dot(elements_col, deltas_row)

        self.weights -= alpha * weight_error
        self.bias -= alpha * (deltas_row)
