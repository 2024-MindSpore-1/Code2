from mindspore import nn, ops


class MAELoss(nn.Cell):
	def construct(self, pred, target):
		return ops.abs(pred - target).mean()


class MAELossTwo(nn.LossBase):
	def construct(self, pred, target):
		return self.get_loss(ops.abs(pred - target))
