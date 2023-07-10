from layer_types.FCLayer import FCLayer
from layer_types.Layer import Layer
from layer_types.ActivationLayer import ActivationLayer
from layer_types.mnist_InputLayer import mnist_InputLayer
from layer_types.SoftmaxLayer import SoftmaxLayer
from layer_types.CrossEntropyLayer import CrossEntropyLayer

from data_structure.Shape import Shape
from data_structure.Tensor import Tensor
from application.SGDTrainer import SGDTrainer
import numpy as np

class Network(object):

    input_layer = None
    layer_list = []
    tensor_list = []
    netsize = 0

    def __init__(self, input_layer, layer_list, tensor_list):
        self.input_layer = input_layer
        self.layer_list = layer_list
        self.tensor_list = tensor_list
        self.netsize = len(self.tensor_list)

    def add_tensor(self, t):
        self.tensor_list.append(t)

    def add_layer(self, l):
        self.layer_list.append(l)

    def forward(self, data):
        self.tensor_list[0].set_elements(self.input_layer.get_train_data(data))
        for i in range(self.netsize - 1):
            self.layer_list[i].forward(self.tensor_list[i], self.tensor_list[i + 1])
        loss = self.layer_list[self.netsize - 1].forward(self.tensor_list[self.netsize - 1], self.input_layer.get_label(data))
        return loss

    def predict(self, data):
        self.tensor_list[0].set_elements(self.input_layer.get_train_data(data))
        for i in range(self.netsize - 1):
            self.layer_list[i].forward(self.tensor_list[i], self.tensor_list[i + 1])
        prediction = self.tensor_list[self.netsize - 1].get_elements()
        return np.argmax(prediction)

    def backprop(self, data, alpha):
        self.forward(data)

        self.layer_list[self.netsize - 1].backward(self.tensor_list[self.netsize - 1], self.input_layer.get_label(data))
        for i in range((self.netsize - 1), 1, -1):
            self.layer_list[i - 1].backward(self.tensor_list[i], self.tensor_list[i - 1])

        for i in range(self.netsize):
            if type(self.layer_list[i]).__name__ == "FCLayer":
                self.layer_list[i].calculate_delta_weights(self.tensor_list[i], self.tensor_list[i + 1], alpha)
