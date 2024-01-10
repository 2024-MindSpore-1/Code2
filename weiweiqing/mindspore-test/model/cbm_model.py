from senet import senet154
from mindspore import nn, ops
import mindspore as ms


class ConvBlock(nn.Cell):
	"""Basic convolutional block.

	convolution + batch normalization + relu.
	Args:
		in_c (int): number of input channels.
		out_c (int): number of output channels.
		k (int or tuple): kernel size.
		s (int or tuple): stride.
		p (int or tuple): padding.
	"""

	def __init__(self, in_c, out_c, k, s=1, p=0):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, pad_mode="pad")
		self.bn = nn.BatchNorm2d(out_c)

	def construct(self, x):
		return ops.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Cell):

	def __init__(self, in_planes, ratio=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.fc = nn.SequentialCell([
			nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
			nn.ReLU(),
			nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
		])
		self.sigmoid = nn.Sigmoid()

	def construct(self, x):
		avg_out = self.fc(self.avg_pool(x))
		max_out = self.fc(self.max_pool(x))
		out = avg_out + max_out
		return self.sigmoid(out)


class SpatialAttention(nn.Cell):
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()

		self.conv1 = nn.Conv2d(
			2,
			1,
			kernel_size,
			padding=kernel_size // 2,
			bias=False)
		self.sigmoid = nn.Sigmoid()

	def construct(self, x):
		avg_out = ops.mean(x, axis=1, keep_dims=True)
		max_out, _ = ops.max(x, axis=1, keepdims=True)
		x = ops.cat([avg_out, max_out], axis=1)
		x = self.conv1(x)
		return self.sigmoid(x)


class BasicBlock(nn.Cell):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
		self.bn2 = nn.BatchNorm2d(planes)

		self.ca = ChannelAttention(planes)
		self.sa = SpatialAttention()

		self.downsample = downsample
		self.stride = stride

	def construct(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out = self.ca(out) * out
		out = self.sa(out) * out

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class CBAM_Block(nn.Cell):
	def __init__(self, planes):
		super(CBAM_Block, self).__init__()
		self.ca = ChannelAttention(planes)
		self.sa = SpatialAttention()

	def construct(self, x):
		return self.sa(self.ca(x))


class MODEL_CBAM(nn.Cell):
	'''
	the mian model of TII2020
	'''

	def __init__(
			self,
			num_classes,
			senet154_weight,
			nchannels=[256, 512, 1024, 2048],
			multi_scale=False,
			use_gpu=True
	):
		super(MODEL_CBAM, self).__init__()
		self.use_gpu = use_gpu
		self.conv = ConvBlock(3, 32, 3, s=2, p=1)
		self.senet154_weight = senet154_weight
		self.num_classes = num_classes

		# construct SEnet154
		senet154_ = senet154(num_classes=1000, pretrained=None)
		# senet154_.load_state_dict(torch.load(self.senet154_weight))
		# param_dict = ms.load_checkpoint(self.senet154_weight)
		# ms.load_param_into_net(senet154_, param_dict)
		self.extract_feature = senet154_.layer0
		#global backbone
		self.global_layer1 = senet154_.layer1
		self.global_layer2 = senet154_.layer2
		self.global_layer3 = senet154_.layer3
		self.global_layer4 = senet154_.layer4

		self.global_out = nn.Dense(2048, num_classes)  # global 分类

		self.ha1 = CBAM_Block(nchannels[0])
		self.ha2 = CBAM_Block(nchannels[1])
		self.ha3 = CBAM_Block(nchannels[2])
		self.ha4 = CBAM_Block(nchannels[3])

		self.dropout = nn.Dropout(p=0.2)  #  分类层之前使用dropout

	def construct(self, x):
		x = self.extract_feature(
			x)  # output shape is 128 * 112 *112  senet154第0层layer 提取特征
		x1 = self.global_layer1(x)  # the output shape is 256*56*56

		x2 = self.global_layer2(x1)  # x2 is 512*28*28
		x2_attn = self.ha2(x2)
		x2_out = x2 * x2_attn

		x3 = self.global_layer3(x2_out)  # x3 is 1024*14*14
		x3_attn = self.ha3(x3)
		x3_out = x3 * x3_attn

		x4 = self.global_layer4(x3_out)  # 2048*7*7
		x4_attn = self.ha4(x4)
		x4_out = x4 * x4_attn

		x4_avg = ops.avg_pool2d(
			x4_out,
			x4_out.size()[2:]).view(x4_out.size(0), -1)  # 全局pooling 2048 之前已经relu过了

		global_out = self.global_out(
			self.dropout(x4_avg))  # 2048 -> num_classes

		return global_out
