import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore import Tensor, Parameter


class Net(nn.Cell):
	def __init__(self):
		super().__init__()
		self.conv = nn.Conv2d(1, 6, 5, pad_mode="valid")
		self.param = Parameter(Tensor(np.array([1.0], np.float32)), "param")

	def construct(self, x):
		x = self.conv(x)
		x = x * self.param
		out = ops.matmul(x, x)
		return out


net = Net()


# region custom trainable parameters

#
# for param in net.get_parameters():
# 	if "conv" in param.name:
# 		param.requires_grad = False
#
# optim = nn.Adam(params=net.trainable_params())
# print(optim.parameters)

# endregion


# region Learning rate -- Dynamic LR

# milestone = [1, 3, 10]
# learning_rates = [0.1, 0.05, 0.01]
#
# lr = nn.piecewise_constant_lr(milestone=milestone, learning_rates=learning_rates)
# print(lr)
#
# optim = nn.SGD(net.trainable_params(), learning_rate=lr)


# endregion

# region Learning rate -- LearningRateSchedule
#
# start_learning_rate = 0.1
# decay_rate = 0.9
# decay_steps = 4
#
# exponential_decay_lr = nn.ExponentialDecayLR(learning_rate=start_learning_rate, decay_rate=decay_rate, decay_steps=decay_steps)

# for i in range(decay_steps):
# 	step = Tensor(i)
# 	lr = exponential_decay_lr(step)
# 	print(f"step: {i + 1}  lr: {lr}")

# optim = nn.Momentum(net.trainable_params(), learning_rate=exponential_decay_lr)


# endregion

# region weight decay

# fixed
# optim = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9, weight_decay=0.9)


# dynamic
# class ExponentialWeightDecay(nn.Cell):
# 	def __init__(self, weight_decay, decay_rate, decay_steps):
# 		super().__init__()
# 		self.weight_decay = weight_decay
# 		self.decay_rate = decay_rate
# 		self.decay_steps = decay_steps
#
# 	def construct(self, global_step):
# 		p = global_step / self.decay_steps
# 		return self.weight_decay * ops.pow(self.decay_rate, p)
#
#
# weight_decay = ExponentialWeightDecay(weight_decay=0.0001, decay_rate=0.1, decay_steps=10000)
# optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9, weight_decay=weight_decay)

# endregion

# region parameter grouping

conv_params = [param for param in net.get_parameters() if "conv" in param.name]
no_conv_params = list(filter(lambda x: "conv" not in x.name, net.get_parameters()))

fix_lr = 0.01

polynomial_decay_lr = nn.PolynomialDecayLR(
	learning_rate=0.1,
	end_learning_rate=0.01,
	decay_steps=4,
	power=0.5
)

params_group = [
	{"params": conv_params, "weight_decay": 0.01, "lr": fix_lr},
	{"params": no_conv_params, "lr": polynomial_decay_lr}
]

optim = nn.Momentum(params=params_group, learning_rate=0.1, momentum=0.9, weight_decay=0.0)

print(optim.group_lr[0].learning_rate.value())

# endregion
