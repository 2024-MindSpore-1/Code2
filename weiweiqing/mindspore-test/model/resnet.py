# ------- from mindspore tutorial ------------------

from mindspore import nn
from mindspore.common.initializer import Normal

from typing import Optional, Type, Union

weight_init = Normal(mean=0, sigma=0.2)
gamma_init = Normal(mean=1, sigma=0.02)


# building block
class ResidualBlockBase(nn.Cell):
	expansion = 1

	def __init__(
			self,
			in_channels: int,
			mid_channels: int,
			out_channels: Optional[int] = None,
			stride: int = 1,
			down_sample: Optional[nn.Cell] = None,
	):
		super().__init__()

		out_channels = out_channels or mid_channels

		self.conv1 = nn.Conv2d(
			in_channels=in_channels,
			out_channels=mid_channels,
			kernel_size=3,
			stride=stride,
			weight_init=weight_init,
		)
		self.norm1 = nn.BatchNorm2d(mid_channels)
		self.conv2 = nn.Conv2d(
			in_channels=mid_channels,
			out_channels=out_channels,
			kernel_size=3,
			weight_init=weight_init,
		)
		self.norm2 = nn.BatchNorm2d(out_channels)

		self.relu = nn.ReLU()

		self.down_sample = down_sample

		# self.down_sample = None
		# if stride > 1 or in_channels != out_channels:
		# 	self.down_sample = nn.Conv2d(
		# 		in_channels=in_channels,
		# 		out_channels=out_channels,
		# 		kernel_size=1,
		# 		stride=stride,
		# 		weight_init=weight_init
		# 	)
		# 	self.norm3 = nn.BatchNorm2d(out_channels, gamma_init=gamma_init)

	def construct(self, x):
		identity = x

		out = self.conv1(x)
		out = self.norm1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.norm2(out)

		# if self.down_sample is not None:
		# 	identity = self.down_sample(x)
		# 	identity = self.norm3(identity)

		if self.down_sample is not None:
			identity = self.down_sample(x)

		out += identity
		out = self.relu(out)

		return out


# Bottleneck
class ResidualBlock(nn.Cell):
	expansion = 4

	def __init__(
			self,
			in_channels: int,
			mid_channels: int,
			out_channels: Optional[int] = None,
			stride: int = 1,
			down_sample: Optional[nn.Cell] = None,
	):
		super().__init__()

		out_channels = out_channels or mid_channels * self.expansion

		self.conv1 = nn.Conv2d(
			in_channels=in_channels,
			out_channels=mid_channels,
			kernel_size=1,
			weight_init=weight_init,
		)
		self.norm1 = nn.BatchNorm2d(mid_channels)

		self.conv2 = nn.Conv2d(
			in_channels=mid_channels,
			out_channels=mid_channels,
			kernel_size=3,
			stride=stride,
			weight_init=weight_init
		)
		self.norm2 = nn.BatchNorm2d(mid_channels)

		self.conv3 = nn.Conv2d(
			in_channels=mid_channels,
			out_channels=out_channels,
			kernel_size=1,
			weight_init=weight_init
		)
		self.norm3 = nn.BatchNorm2d(out_channels)

		self.down_sample = down_sample
		# self.down_sample = None
		# if stride > 1 or in_channels != out_channels:
		# 	self.down_sample = nn.Conv2d(
		# 		in_channels=in_channels,
		# 		out_channels=out_channels,
		# 		kernel_size=1,
		# 		stride=stride,
		# 		weight_init=weight_init
		# 	)
		# 	self.norm4 = nn.BatchNorm2d(out_channels, gamma_init=gamma_init)

		self.relu = nn.ReLU()

	def construct(self, x):
		identity = x

		out = self.conv1(x)
		out = self.norm1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.norm2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.norm3(out)

		# if self.down_sample is not None:
		# 	identity = self.down_sample(x)
		# 	identity = self.norm4(identity)

		if self.down_sample is not None:
			identity = self.down_sample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Cell):
	def __init__(
			self,
			num_classes: int,
			nums_layer: list,
			block: Type[Union[ResidualBlockBase, ResidualBlock]],
	):
		super().__init__()
		weight_init = Normal(mean=0, sigma=0.2)
		self.relu = nn.ReLU()

		# 3 * 224*224
		# head
		self.conv1 = nn.Conv2d(
			in_channels=3,
			out_channels=64,
			kernel_size=7,
			stride=2,
			weight_init=weight_init
		)
		# 64 * 112*112
		self.norm = nn.BatchNorm2d(64)
		self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

		# 64 * 56*56

		self.layer1, out_channels = self._make_layer(
			in_channels=64,
			mid_channels=64,
			block=block,
			nums_block=nums_layer[0],
		)
		# 256 * 28*28
		self.layer2, out_channels = self._make_layer(
			in_channels=out_channels,
			mid_channels=128,
			block=block,
			nums_block=nums_layer[1],
			stride=2,
		)  # 512
		self.layer3, out_channels = self._make_layer(
			in_channels=out_channels,
			mid_channels=256,
			block=block,
			nums_block=nums_layer[2],
			stride=2,
		)  # 1024
		self.layer4, out_channels = self._make_layer(
			in_channels=out_channels,
			mid_channels=512,
			block=block,
			nums_block=nums_layer[3],
			stride=2
		)  # 2048

		# tail
		self.avg_pool = nn.AvgPool2d(kernel_size=7, pad_mode="pad")
		self.flatten = nn.Flatten()
		self.fc = nn.Dense(
			in_channels=out_channels,
			out_channels=num_classes,
		)

	def construct(self, x):
		x = self.conv1(x)
		x = self.norm(x)
		x = self.relu(x)
		x = self.max_pool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avg_pool(x)
		x = self.flatten(x)
		x = self.fc(x)

		return x

	@staticmethod
	def _make_layer(
			in_channels: int,
			mid_channels: int,
			block: Type[Union[ResidualBlockBase, ResidualBlock]],
			nums_block: int,
			stride: int = 1,
	):
		down_sample = None  # shortcuts分支
		if stride != 1 or in_channels != mid_channels * block.expansion:

			down_sample = nn.SequentialCell([
				nn.Conv2d(in_channels, mid_channels * block.expansion, kernel_size=1, stride=stride, weight_init=weight_init),
				nn.BatchNorm2d(mid_channels * block.expansion, gamma_init=gamma_init)
			])

		layers = [block(in_channels, mid_channels, stride=stride, down_sample=down_sample)]

		in_channels = mid_channels * block.expansion if block == ResidualBlock else mid_channels
		for _ in range(1, nums_block):
			layers.append(block(in_channels, mid_channels))

		return nn.SequentialCell(layers), in_channels

	@staticmethod
	def get_resnet18(num_classes):
		nums_layer = [2, 2, 2, 2]
		return ResNet(num_classes=num_classes, nums_layer=nums_layer, block=ResidualBlockBase)

	@staticmethod
	def get_resnet34(num_classes):
		nums_layer = [3, 4, 6, 3]
		return ResNet(num_classes=num_classes, nums_layer=nums_layer, block=ResidualBlockBase)

	@staticmethod
	def get_resnet50(num_classes):
		nums_layer = [3, 4, 6, 3]
		return ResNet(num_classes=num_classes, nums_layer=nums_layer, block=ResidualBlock)

	@staticmethod
	def get_resnet101(num_classes):
		nums_layer = [3, 4, 23, 3]
		return ResNet(num_classes=num_classes, nums_layer=nums_layer, block=ResidualBlock)

	@staticmethod
	def get_resnet152(num_classes):
		nums_layer = [3, 8, 36, 3]
		return ResNet(num_classes=num_classes, nums_layer=nums_layer, block=ResidualBlock)
