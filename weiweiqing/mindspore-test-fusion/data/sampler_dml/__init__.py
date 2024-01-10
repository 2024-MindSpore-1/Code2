
_dict = {}


def register_data_sampler(cls):
	if cls.__name__ in _dict:
		raise ValueError("Cannot register duplicated data sampler: " + cls.__name__)
	_dict[cls.__name__] = cls
	return cls


def get_data_sampler(data_sampler_name, opt, image_dict, image_list):
	if data_sampler_name in _dict:
		return _dict[data_sampler_name](opt, image_dict, image_list)
	raise ValueError("Cannot find data sampler name:" + data_sampler_name)


from .class_random_sampler import ClassRandomSampler
