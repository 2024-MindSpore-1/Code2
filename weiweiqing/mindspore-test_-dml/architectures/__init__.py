_dict = {}


def register_arc(cls):
	if cls.__name__ in _dict:
		raise ValueError("Cannot register duplicated architecture: {}".format(cls.__name__))
	_dict[cls.__name__] = cls
	return cls


def get_arch(arch_name, **kwargs):
	if arch_name in _dict:
		return _dict[arch_name](**kwargs)

	raise ValueError("Cannot find architecture name: " + arch_name)


from .vit import Vit
from .resnet50 import Resnet50
