from . import SimpleTrain
from mindspore.amp import build_train_network, FixedLossScaleManager, DynamicLossScaleManager


class AMPGpuTrain(SimpleTrain):
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
			lr_scheduler=None
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
		loss_scale_manager = FixedLossScaleManager()
		self.net = build_train_network(self.net, self.optimizer, self.loss_fn, level="O2", loss_scale_manager=loss_scale_manager)
