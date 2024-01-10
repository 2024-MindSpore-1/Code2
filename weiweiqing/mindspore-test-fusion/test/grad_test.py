import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
import math

# region simple example

# x = ops.ones(5, ms.float32)
# y = ops.zeros(3, ms.float32)
# w = Parameter(name="w", default_input=Tensor(np.random.randn(5, 3), ms.float32))
# b = Parameter(name="b", default_input=Tensor(np.random.randn(3), ms.float32))
#
#
# def function(x, y, w, b):
# 	z = ops.matmul(x, w) + b
# 	loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
# 	return loss
#
#
# loss = function(x, y, w, b)
# print(loss)
#
# grad_fn = ms.grad(function, (2, 3))
#
# grads = grad_fn(x, y, w, b)
# print(grads)


# endregion

# region first order derivation

# x = Tensor(3.0, dtype=ms.float32)
# y = Tensor(5.0, dtype=ms.float32)
#
#
# class Net(nn.Cell):
# 	def __init__(self):
# 		super().__init__()
# 		self.z = Parameter(Tensor(1.0, ms.float32), name="z")
#
# 	def construct(self, x, y):
# 		return x * x * y * self.z
#
#
# net = Net()


# derive x and y
# grad_fn = ms.grad(net, grad_position=(0, 1))
#
# gradients = grad_fn(x, y)
# print(gradients)

# derive z
# grad_fn = ms.grad(net, grad_position=None, weights=net.trainable_params())
# gradients = grad_fn(x, y)
# print(gradients)

# derive x, y and z
# grad_fn = ms.grad(net, grad_position=(0, 1), weights=net.trainable_params())
# gradients, out = grad_fn(x, y)
# print(gradients)
# print(out)

# endregion


# region high-order deviation

# single input and single output

# class Net(nn.Cell):
# 	def __init__(self):
# 		super().__init__()
# 		self.sin = ops.sin
#
# 	def construct(self, x):
# 		return self.sin(x)
#
#
# net = Net()
#
# first_grad = ms.grad(net)
# second_grad = ms.grad(first_grad)
#
# input = Tensor(math.pi, ms.float32)
# output = np.round(second_grad(input).asnumpy(), 2)
# print(output)

# single input and multiple outputs
# class Net(nn.Cell):
# 	def __init__(self):
# 		super().__init__()
# 		self.sin = ops.sin
# 		self.cos = ops.cos
#
# 	def construct(self, x):
# 		out1 = self.sin(x)
# 		out2 = self.cos(x)
# 		return out1, out2
#
#
# net = Net()
#
# first_grad = ms.grad(net)
# second_grad = ms.grad(first_grad)
#
# input = Tensor(math.pi, ms.float32)
# output = np.round(second_grad(input).asnumpy(), 2)
# print(output)

# multiple inputs and multiple outputs
class Net(nn.Cell):
	def __init__(self):
		super().__init__()

	def construct(self, x, y):
		out1 = ops.sin(x) - ops.cos(y)
		out2 = ops.cos(x) - ops.sin(y)
		return out1, out2


net = Net()
first_grad = ms.grad(net, grad_position=(0, 1))
second_grad = ms.grad(first_grad, grad_position=(0, 1))

x = Tensor(math.pi, ms.float32)
y = Tensor(math.pi, ms.float32)

gradients = second_grad(x, y)
print(gradients)

# endregion
