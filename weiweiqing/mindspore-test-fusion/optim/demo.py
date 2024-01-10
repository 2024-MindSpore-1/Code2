from mindspore import nn, Tensor, Parameter, ops
import mindspore as ms


class Momentum(nn.Optimizer):
	def __init__(self, params, learning_rate, momentum=0.9):
		super().__init__(parameters=params, learning_rate=learning_rate)
		self.momentum = Parameter(Tensor(momentum, ms.float32), "momentum")
		self.moments = self.parameters.clone(prefix="moments", init="zeros")

	def construct(self, gradients):
		lr = self.get_lr()
		params = self.parameters

		for i in range(len(params)):
			ops.assign(self.moments[i], self.moments[i] * self.momentum + gradients[i])
			update = params[i] - lr * self.momentum[i]
			ops.assign(params[i], update)
		return params
