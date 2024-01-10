import numpy as np
import mindspore
from mindspore import nn, Tensor, Parameter, ops


class Net(nn.Cell):
	def __init__(self):
		super().__init__()
		self.w = Parameter(Tensor(np.random.randn(5, 3), mindspore.float32), "w")  # weights
		self.b = Parameter(Tensor(np.random.randn(3,), mindspore.float32), "b") # bias

	def construct(self, x):
		return ops.matmul(x, self.w) + self.b

net = Net()

# print(net.w)
# print(net.b)

# print(net.w.asnumpy())
# print(net.b.asnumpy())

# for parameter in net.get_parameters():
# 	print(parameter)

# for parameter in net.trainable_params():
# 	print(parameter)

# for parameter in net.parameters_and_names():
# 	print(parameter)

# print(net.parameters_dict()["w"])
# print(net.parameters_dict()["b"])

print(net.b.asnumpy())
net.b.set_data(Tensor([2, 3, 4]))
print(net.b.asnumpy())
