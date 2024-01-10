from mindspore import nn, ops, Tensor, dtype


class CrossEntropySmooth(nn.LossBase):
	def __init__(
			self,
			num_classes,
			sparse=True,
			reduction="mean",
			smooth_factor=0.0,
	):
		super().__init__()
		self.sparse = sparse
		self.on_value = Tensor(1 - smooth_factor, dtype.float32)
		self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), dtype.float32)

		self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

	def construct(self, logits, labels):
		if self.sparse:
			labels = ops.one_hot(labels, logits.shape[1], self.on_value, self.off_value)
		loss = self.ce(logits, labels)
		return loss
