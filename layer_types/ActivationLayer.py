from layer_types.Layer import Layer
import numpy as np

class ActivationLayer(Layer):

    def __init__(self, inTensor, outTensor):
        self.inTensor = inTensor
        self.outTensor = outTensor

    def forward(self, inTensor, outTensor):
        x = self.inTensor.get_elements()
        outTensor.set_elements(( 1/(1 + np.exp(-(x)))))
        return outTensor

    def backward(self, outTensor, inTensor):
        elements = outTensor.get_elements()
        deltas = outTensor.get_deltas()
        inTensor.set_deltas(((elements * (1 - elements)) * deltas))
        return inTensor