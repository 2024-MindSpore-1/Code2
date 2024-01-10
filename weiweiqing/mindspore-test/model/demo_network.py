# ------- from mindspore tutorial ------------------

from mindspore import nn
import mindspore as ms


class DemoNetwork(nn.Cell):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.dense_relu_sequential = nn.SequentialCell(
			nn.Dense(28*28, 512),
			nn.ReLU(),
			nn.Dense(512, 512),
			nn.ReLU(),
			nn.Dense(512, 10)
		)

	def construct(self, x):
		x = self.flatten(x)
		logits = self.dense_relu_sequential(x)
		return logits



