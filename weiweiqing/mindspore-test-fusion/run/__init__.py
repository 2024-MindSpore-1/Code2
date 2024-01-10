import mindspore as ms
import os
import stat
from mindspore.common.initializer import initializer, Normal
import numpy as np


class RunUtil:

	@staticmethod
	def remove_ckpt_file(net, path, accuracy, epoch):
		file_name = f"{net.__class__.__name__}_{(accuracy*100):.2f}_{epoch}.ckpt"
		full_path = os.path.join(path, file_name)
		if os.path.exists(full_path):
			os.chmod(full_path, stat.S_IWRITE)
			os.remove(full_path)

	@staticmethod
	def save_net(net, path, accuracy, epoch):
		file_name = f"{net.__class__.__name__}_{(accuracy*100):.2f}_{epoch}.ckpt"
		full_path = os.path.join(path, file_name)
		ms.save_checkpoint(net, full_path)
		return full_path

	@staticmethod
	def load_net(net, path):
		param_dict = ms.load_checkpoint(path)
		param_not_load, _ = ms.load_param_into_net(net, param_dict)

	@staticmethod
	def init_parameters(net):
		for name, param in net.parameters_and_names():
			if 'weight' in name:
				param.set_data(initializer(Normal(), param.shape, param.dtype))
			if 'bias' in name:
				param.set_data(initializer('zeros', param.shape, param.dtype))

	@staticmethod
	def get_device_type():
		return ms.get_context("device_target")

	@staticmethod
	def set_graph_mode():
		# ms.set_context(mode=ms.PYNATIVE_MODE)
		ms.set_context(mode=ms.GRAPH_MODE)

	@staticmethod
	def get_train_engine():
		dev_type = RunUtil.get_device_type().title()
		print(f"\n------- Run with [{ dev_type }] device -----------")
		return AMPGpuTrain if dev_type == "Gpu" or dev_type == "Ascend" else AMPTrain

	@staticmethod
	def ignore_warnings():
		import warnings
		warnings.filterwarnings("ignore")


from .base import SimpleTrain, SimpleEvaluation, AMPTrain, AMPGpuTrain
