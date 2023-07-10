from application.Network import Network
from data_structure.Tensor import Tensor
from data_structure.Shape import Shape

import numpy as np


class Metric:

    def __init__(self):
        'constructor'

    def predict(self, network, data):
        predictor = data[0]
        label = data[1]
        positive = 0
        negative = 0

        for j in range(len(predictor)):
            single_data = [predictor[j], label[j]]
            prediction = network.predict(single_data)
            if label[j] == prediction:
                positive += 1
            else: negative += 1

        return positive/(positive + negative)



