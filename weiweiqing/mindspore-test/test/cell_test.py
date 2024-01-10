from mindspore import ops, nn, Tensor
import mindspore as ms


# class Threshold(nn.Cell):
# 	def __init__(self, threshold, value):
# 		super().__init__()
# 		self.threshold = threshold
# 		self.value = value
#
# 	def construct(self, x):
# 		mask = ops.gt(x, self.threshold)
# 		value = ops.fill(x.dtype, x.shape, self.value)
# 		return ops.select(mask, x, value)
#
#
# cell = Threshold(0.15, 111)
# x = Tensor([0.1, 0.2, 0.3], ms.float32)
# print(cell(x))


class Dropout2d(nn.Cell):
	def __init__(self, keep_prob):
		super().__init__()
		self.keep_prob = keep_prob
		self.dropout2d = ops.Dropout2D(keep_prob)

	def construct(self, x):
		return self.dropout2d(x)

	def bprop(self, x, out, dout):
		_, mask = out
		dy, _ = dout
		if self.keep_prob != 0:
			dy = dy * (1 / self.keep_prob)
		dy = mask.astype(ms.float32) * dy
		return (dy.astype(x.dtype),)


dropout_2d = Dropout2d(0.8)
dropout_2d.bprop_debug = True
