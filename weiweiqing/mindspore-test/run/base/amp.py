from . import SimpleTrain
from mindspore.amp import auto_mixed_precision, DynamicLossScaler, StaticLossScaler, all_finite
import mindspore as ms


class AMPTrain(SimpleTrain):
	def __init__(
			self,
			train_dataset,
			val_dataset,
			net,
			loss_fn,
			optimizer,
			epochs,
			rel_path="./results",
			initialize_parameters=False,
			lr_scheduler=None,
	):
		super().__init__(
			train_dataset=train_dataset,
			val_dataset=val_dataset,
			net=net,
			loss_fn=loss_fn,
			optimizer=optimizer,
			lr_scheduler=lr_scheduler,
			epochs=epochs,
			rel_path=rel_path,
			initialize_parameters=initialize_parameters,
		)
		# O0 keep FP32, O1 cast to FP16 according to white list
		# O2 keep FP32 according to black list, others cast to FP16
		# O3 cast to FP16 completely
		self.net = auto_mixed_precision(self.net, "O2")
		self.loss_scaler = DynamicLossScaler(scale_value=2**10, scale_factor=2, scale_window=50)

	def forward_fn(self, data, label):
		pred = self.net(data)
		loss = self.loss_fn(pred, label)
		# scale up the loss value
		loss = self.loss_scaler.scale(loss)
		return loss, pred

	@ms.jit
	def train_step(self, data, label):
		(loss, _), grads = self.grad_fn(data, label)
		loss = self.loss_scaler.unscale(loss)

		is_finite = all_finite(grads)
		if is_finite:
			grads = self.loss_scaler.unscale(grads)
			self.optimizer(grads)
		self.loss_scaler.adjust(is_finite)

		return loss
