# The code in the package has been adapted from https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch


_dict = {}


def register_dataset(fn):
	if fn.__name__ in _dict:
		raise ValueError(f"Cannot register duplicated dataset: {fn.__name__}")
	_dict[fn.__name__] = fn
	return fn


def get_datasets(dataset_name, opt, data_path):
	if dataset_name not in _dict:
		raise NotImplementedError(f"A dataset for {dataset_name} is currently not implemented!")

	return _dict[dataset_name](opt, data_path)


from .basic_dataset_scaffold import BaseDataset
from .cars196 import Cars196
from .cub200 import Cub200
from .food import Food
