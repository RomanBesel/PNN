import mlxtend
from mlxtend.data import loadlocal_mnist
import numpy as np
from metrics.Metric import Metric

from layer_types.FCLayer import FCLayer
from layer_types.Layer import Layer
from layer_types.ActivationLayer import ActivationLayer
from layer_types.mnist_InputLayer import mnist_InputLayer
from layer_types.SoftmaxLayer import SoftmaxLayer
from layer_types.CrossEntropyLayer import CrossEntropyLayer

from data_structure.Shape import Shape
from data_structure.Tensor import Tensor

from application.Network import Network
from application.SGDTrainer import SGDTrainer

X_train, y_train = loadlocal_mnist(images_path = r'C:\Users\roman\Programming\mnist\train-images.idx3-ubyte', labels_path = r'C:\Users\roman\Programming\mnist\train-labels.idx1-ubyte')
X_test, y_test = loadlocal_mnist(images_path = r'C:\Users\roman\Programming\mnist\t10k-images.idx3-ubyte', labels_path = r'C:\Users\roman\Programming\mnist\t10k-labels.idx1-ubyte')

print('Dimensions: %s x %s' % (X_train.shape[0], X_train.shape[1]))
#print('\n1st row', X_train[0])

train_data = [X_train, y_train]
test_data = [X_test, y_test]

sh784 = Shape([1, 784])
sh150 = Shape([1, 150])
sh75 = Shape([1, 75])
sh10 = Shape([1, 10])

t1 = Tensor([], sh784, [])
t2 = Tensor([], sh150, [])
t3 = Tensor([], sh150, [])
t4 = Tensor([], sh75, [])
t5 = Tensor([], sh75, [])
t6 = Tensor([], sh10, [])
t7 = Tensor([], sh10, [])

fc1 = FCLayer(t1, t2)
fc2 = FCLayer(t3, t4)
fc3 = FCLayer(t5, t6)

act1 = ActivationLayer(t2, t3)
act2 = ActivationLayer(t4, t5)
soft = SoftmaxLayer(t6, t7)
cross = CrossEntropyLayer(t7)

input_layer = mnist_InputLayer()

layer_list = [fc1, act1, fc2, act2, fc3, soft, cross]
tensor_list = [t1, t2, t3, t4, t5, t6, t7]

network = Network(input_layer, layer_list, tensor_list)
sgd_trainer = SGDTrainer(1, 0.05, 5, False)
metric = Metric()

model = sgd_trainer.optimize(network, train_data)
performance = metric.predict(model, test_data)

print(performance)





