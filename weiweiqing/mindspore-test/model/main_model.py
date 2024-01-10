from mindspore import nn, ops, Tensor
import mindspore as ms
from mindspore.common.initializer import initializer
from model.resnet import ResNet


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


class SpatialAttn(nn.Cell):
	"""
	空间上的attention 模块.
	Spatial Attention (Sec. 3.1.I.1)"""

	def __init__(self):
		super(SpatialAttn, self).__init__()
		self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
		self.conv2 = ConvBlock(1, 1, 1)

	def construct(self, x):
		# global cross-channel averaging
		x = x.mean(1, keep_dims=True)  # 由hwc 变为 hw1
		# 3-by-3 conv
		h = x.shape[2]
		x = self.conv1(x)
		# bilinear resizing
		x = ops.upsample(x, (h, h), mode='bilinear', align_corners=True)
		# scaling conv
		x = self.conv2(x)
		## 返回的是h*w*1 的 soft map
		return x


class ChannelAttn(nn.Cell):
	"""
	通道上的attention 模块
	Channel Attention (Sec. 3.1.I.2)
	"""

	def __init__(self, in_channels, reduction_rate=16):
		super(ChannelAttn, self).__init__()
		assert in_channels % reduction_rate == 0
		self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
		self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

	def construct(self, x):
		# squeeze operation (global average pooling)
		x = ops.avg_pool2d(x, x.shape[2:])  #输出是1*1*c
		# excitation operation (2 conv layers)
		x = self.conv1(x)
		x = self.conv2(x)
		# 返回的是 1*1*c 的map
		return x


class SoftAttn(nn.Cell):
	"""
	Soft Attention (Sec. 3.1.I)
	空间和通道上的attention 融合
	就是空间和通道上的attention做一个矩阵乘法
	Aim: Spatial Attention + Channel Attention

	Output: attention maps with shape identical to input.
	"""

	def __init__(self, in_channels):
		super(SoftAttn, self).__init__()
		self.spatial_attn = SpatialAttn()
		self.channel_attn = ChannelAttn(in_channels)
		self.conv = ConvBlock(in_channels, in_channels, 1)

	def construct(self, x):
		# 返回的是 hwc
		y_spatial = self.spatial_attn(x)
		y_channel = self.channel_attn(x)
		y = y_spatial * y_channel
		y = ops.sigmoid(self.conv(y))
		return y


class HardAttn(nn.Cell):
	"""Hard Attention (Sec. 3.1.II)"""

	def __init__(self, in_channels):
		super(HardAttn, self).__init__()
		self.fc = nn.Dense(in_channels, 4 * 2)
		self.init_params()

	def init_params(self):
		for name, param in self.fc.parameters_and_names():
			if 'weight' in name:
				param.set_data(initializer("zeros", param.shape, param.dtype))
			if 'bias' in name:
				param.set_data(Tensor([0.3, -0.3, 0.3, 0.3, -0.3, 0.3, -0.3, -0.3], dtype=ms.float32))

		# self.fc.weight.data.zero_()
		# # 初始化 参数
		# # if x_t = 0  the performance is very low
		# self.fc.bias.data.copy_(
		# 	Tensor([0.3, -0.3, 0.3, 0.3, -0.3, 0.3, -0.3, -0.3], dtype=ms.float32))

	def construct(self, x):
		'''
		输出的是STN 需要的theta
		'''
		# squeeze operation (global average pooling)
		x = ops.avg_pool2d(x, x.shape[2:]).view(x.shape[0], x.shape[1])
		# predict transformation parameters
		theta = ops.tanh(self.fc(x))
		theta = theta.view(-1, 4, 2)
		#  返回的是 2T  T为区域数量。 因为尺度会固定。 所以只要学位移的值
		return theta


class HarmAttn(nn.Cell):
	"""Harmonious Attention (Sec. 3.1)"""

	def __init__(self, in_channels):
		super(HarmAttn, self).__init__()
		self.soft_attn = SoftAttn(in_channels)
		self.hard_attn = HardAttn(in_channels)

	def construct(self, x):
		y_soft_attn = self.soft_attn(x)
		theta = self.hard_attn(x)
		return y_soft_attn, theta


class SENet_BASE(nn.Cell):

	def __init__(
		self,
		num_classes,
		senet154_weight,
		multi_scale=False,
	):
		super(SENet_BASE, self).__init__()
		self.senet154_weight = senet154_weight
		self.num_classes = num_classes

		# construct SEnet154
		# senet154_ = senet154(num_classes=1000, pretrained=None)
		senet154_ = senet154(num_classes=1000, pretrained=None)
		# senet154_.load_state_dict(torch.load(self.senet154_weight))
		# params_dict = ms.load_checkpoint(self.senet154_weight)
		# ms.load_param_into_net(senet154_, params_dict)
		self.main_network = nn.SequentialCell([
			senet154_.layer0,
			senet154_.layer1,
			senet154_.layer2,
			senet154_.layer3,
			senet154_.layer4,
			nn.AdaptiveAvgPool2d((1, 1))
		])

		self.global_out = nn.SequentialCell([nn.Dropout(p=0.2), nn.Dense(2048, num_classes)])

	def construct(self, x):
		x = self.main_network(x)
		x = x.view(x.shape[0], -1)
		return self.global_out(x)


class MODEL(nn.Cell):
	'''
	the mian model of TII2020
	'''

	def __init__(
			self,
			num_classes,
			senet154_weight,
			nchannels=[256, 512, 1024, 2048],
			multi_scale=False):
		super(MODEL, self).__init__()
		self.senet154_weight = senet154_weight
		self.multi_scale = multi_scale
		self.num_classes = num_classes

		# construct SEnet154
		senet154_ = senet154(num_classes=num_classes, pretrained=None)
		# senet154_.load_state_dict(torch.load(self.senet154_weight))
		params_dict = ms.load_checkpoint(self.senet154_weight)
		ms.load_param_into_net(senet154_, params_dict)

		self.global_layers = nn.SequentialCell([
			senet154_.layer0,
			senet154_.layer1,
			senet154_.layer2,
			senet154_.layer3,
			senet154_.layer4,
		])

		self.ha_layers = nn.SequentialCell([
			HarmAttn(nchannels[1]),
			HarmAttn(nchannels[2]),
			HarmAttn(nchannels[3]),
		])

		if self.multi_scale:
			self.global_out = nn.Dense(nchannels[1] + nchannels[2] + nchannels[3], num_classes)
		else:
			self.global_out = nn.SequentialCell([nn.Dropout(p=0.2), nn.Dense(2048, num_classes)])

	def construct(self, x):
		"""
		add the mutli-scal method 10/17.
		first how is the fusion global and local recognition performance
		second add the multi-scal method
		where added? oral output or attention's output?
		"""

		def msa_block(inp, main_conv, harm_attn):
			inp = main_conv(inp)
			inp_attn, _ = harm_attn(inp)
			return inp_attn * inp

		x = self.global_layers[0](x)
		x1 = self.global_layers[1](x)
		# 512*28*28
		x2 = msa_block(x1, self.global_layers[2], self.ha_layers[0])
		# 1024*14*14
		x3 = msa_block(x2, self.global_layers[3], self.ha_layers[1])
		# 2048*7*7
		x4 = msa_block(x3, self.global_layers[4], self.ha_layers[2])

		#全局pooling 2048
		x4_avg = ops.avg_pool2d(x4, x4.shape[2:]).view(x4.shape[0], -1)

		if self.multi_scale:
			#512 向量
			x2_avg = ops.adaptive_avg_pool2d(x2, (1, 1)).view(x2.shape[0], -1)
			#1024 向量
			x3_avg = ops.adaptive_avg_pool2d(x3, (1, 1)).view(x3.shape[0], -1)

			multi_scale_feature = ops.cat([x2_avg, x3_avg, x4_avg], 1)
			global_out = self.global_out(multi_scale_feature)

		else:
			# global_out = self.global_out(
			# 	self.dropout(x4_avg))  # 2048 -> num_classes
			global_out = self.global_out(x4_avg)  # 2048 -> num_classes

		return global_out


class MODEL_Another(nn.Cell):
	'''
	the main model of TII2020
	'''

	def __init__(
			self,
			num_classes,
			resnet50_weight,
			nchannels=[256, 512, 1024, 2048],
			multi_scale=False):
		super(MODEL_Another, self).__init__()
		self.resnet50_weight = resnet50_weight
		self.multi_scale = multi_scale
		self.num_classes = num_classes

		# construct resnet50
		resnet50_ = ResNet.get_resnet50(num_classes=1000)
		params_dict = ms.load_checkpoint(self.resnet50_weight)
		ms.load_param_into_net(resnet50_, params_dict, strict_load=True)

		resnet50_.layer0 = nn.SequentialCell([
			resnet50_.conv1,
			resnet50_.norm,
			resnet50_.relu,
			resnet50_.max_pool
		])

		self.global_layers = nn.SequentialCell([
			resnet50_.layer0,
			resnet50_.layer1,
			resnet50_.layer2,
			resnet50_.layer3,
			resnet50_.layer4,
		])

		self.ha_layers = nn.SequentialCell([
			HarmAttn(nchannels[1]),
			HarmAttn(nchannels[2]),
			HarmAttn(nchannels[3]),
		])

		if self.multi_scale:
			self.global_out = nn.Dense(nchannels[1] + nchannels[2] + nchannels[3], num_classes)
		else:
			self.global_out = nn.SequentialCell([nn.Dropout(p=0.2), nn.Dense(2048, num_classes)])

	def construct(self, x):
		"""
		add the mutli-scal method 10/17.
		first how is the fusion global and local recognition performance
		second add the multi-scal method
		where added? oral output or attention's output?
		"""

		def msa_block(inp, main_conv, harm_attn):
			inp = main_conv(inp)
			inp_attn, _ = harm_attn(inp)
			return inp_attn * inp

		x = self.global_layers[0](x)
		x1 = self.global_layers[1](x)
		# 512*28*28
		x2 = msa_block(x1, self.global_layers[2], self.ha_layers[0])
		# 1024*14*14
		x3 = msa_block(x2, self.global_layers[3], self.ha_layers[1])
		# 2048*7*7
		x4 = msa_block(x3, self.global_layers[4], self.ha_layers[2])

		#全局pooling 2048
		x4_avg = ops.avg_pool2d(x4, x4.shape[2:]).view(x4.shape[0], -1)

		if self.multi_scale:
			#512 向量
			x2_avg = ops.adaptive_avg_pool2d(x2, (1, 1)).view(x2.shape[0], -1)
			#1024 向量
			x3_avg = ops.adaptive_avg_pool2d(x3, (1, 1)).view(x3.shape[0], -1)

			multi_scale_feature = ops.cat([x2_avg, x3_avg, x4_avg], 1)
			global_out = self.global_out(multi_scale_feature)

		else:
			# global_out = self.global_out(
			# 	self.dropout(x4_avg))  # 2048 -> num_classes
			global_out = self.global_out(x4_avg)  # 2048 -> num_classes

		return global_out
