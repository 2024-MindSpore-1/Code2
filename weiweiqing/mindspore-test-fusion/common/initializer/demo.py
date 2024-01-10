import math
import numpy as np
from mindspore.common.initializer import Initializer


class XavierNormal(Initializer):
	def __init__(self, gain=1):
		super().__init__()
		# 配置初始化所需要的参数
		self.gain = gain

	def _initialize(self, arr):  # arr为需要初始化的Tensor
		fan_in, fan_out = self._calculate_fan_in_and_fan_out(arr)  # 计算fan_in, fan_out值

		std = self.gain * math.sqrt(2.0 / float(fan_in + fan_out))  # 根据公式计算std值
		data = np.random.normal(0, std, arr.shape)  # 使用numpy构造初始化好的ndarray

		arr[:] = data[:]  # 将初始化好的ndarray赋值到arr

	@staticmethod
	def _calculate_fan_in_and_fan_out(arr):
		# 计算fan_in和fan_out。fan_in是 `arr` 中输入单元的数量，fan_out是 `arr` 中输出单元的数量。
		shape = arr.shape
		dimensions = len(shape)
		if dimensions < 2:
			raise ValueError("'fan_in' and 'fan_out' can not be computed for arr with fewer than"
							 " 2 dimensions, but got dimensions {}.".format(dimensions))
		if dimensions == 2:  # Linear
			fan_in = shape[1]
			fan_out = shape[0]
		else:
			num_input_fmaps = shape[1]
			num_output_fmaps = shape[0]
			receptive_field_size = 1
			for i in range(2, dimensions):
				receptive_field_size *= shape[i]
			fan_in = num_input_fmaps * receptive_field_size
			fan_out = num_output_fmaps * receptive_field_size
		return fan_in, fan_out
