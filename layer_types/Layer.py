class Layer():

    def __init__(self):
        print('layer constructed')

    def forward(self, inTensor):
            raise NotImplementedError

    def backward(self, outTensor_error, alpha):
            raise NotImplementedError

    def calculate_delta_weights(outTensor):
            raise NotImplementedError


