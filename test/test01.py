

from data_structure.Tensor import Tensor
from data_structure.Shape import Shape

from layer_types.Layer import Layer
from layer_types.FCLayer import FCLayer
from layer_types.InputLayer import InputLayer
from layer_types.ActivationLayer import ActivationLayer
from layer_types.SoftmaxLayer import SoftmaxLayer
from layer_types.CrossEntropyLayer import CrossEntropyLayer


sh12 = Shape([1, 2])
sh13 = Shape([1, 3])

t1 = Tensor([0.4183, 0.5209, 0.0291], sh13, [])
t2 = Tensor([0, 0, 0], sh13, [])
t3 = Tensor([0, 0, 0], sh13, [])
t4 = Tensor([0, 0, 0], sh12, [])
t5 = Tensor([0, 0, 0], sh12, [])

fc1 = FCLayer(t1, t2)
fc1.set_weights([[-0.5057, 0.3987, -0.8943],[0.3356, 0.1673, 0.8321],[-0.3485, -0.4597, -0.1121]])
fc1.set_bias([0.0, 0.0, 0.0])

fc2 = FCLayer(t3, t4)
fc2.set_weights([[0.4047, 0.9563],[-0.8192, -0.1274],[0.3662, -0.7252]])
fc2.set_bias([0.0, 0.0])

act1 = ActivationLayer()
soft1 = SoftmaxLayer()
loss1 = CrossEntropyLayer()

t2.set_elements(fc1.forward(t1))
t3.set_elements(act1.forward(t2))
t4.set_elements(fc2.forward(t3))
t5.set_elements(soft1.forward(t4))

print('\n')
print('forward path')
print(t1.get_elements())
print(t2.get_elements())
print(t3.get_elements())
print(t4.get_elements())
print(t5.get_elements())
print(loss1.forward(t5, [0.7095, 0.0942]))
print('\n')

t5.set_deltas(loss1.backward(t5, [0.7095, 0.0942]))
t4.set_deltas(soft1.backward(t5, t4))
t3.set_deltas(fc2.backward(t4))
t2.set_deltas(act1.backward(t3))

print('backward path')
print(t5.get_deltas())
print(t4.get_deltas())
print(t3.get_deltas())
print(t2.get_deltas())
print('\n')

print('original weight matrix')
print(fc1.get_weights())
print(fc2.get_weights())
print('\n')

fc1.calculate_delta_weights(t1, t2, 1)
fc2.calculate_delta_weights(t3, t4, 1)

print('\n')
print('updated weight matrix fc1')
print(fc1.get_weights())
print('\n')
print('updated weight matrix fc1')
print(fc2.get_weights())
print('\n')
print('bias fc1')
print(fc1.get_bias())
print('\n')
print('bias fc1')
print(fc2.get_bias())





