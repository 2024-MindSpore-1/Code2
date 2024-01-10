from mindspore.common.initializer import initializer
from mindspore import Parameter


class ModelUtil:
	@staticmethod
	def create_parameter(init_type, shape, dtype, name, requires_grad):
		initial = initializer(init_type, shape, dtype).init_data()
		return Parameter(initial, name=name, requires_grad=requires_grad)


from .demo_network import DemoNetwork
from .resnet import ResNet
from .vit import ViT
