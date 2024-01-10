import mindspore.dataset as ds
import numpy as np


class MySampler(ds.Sampler):
	def __init__(self):
		super().__init__()
		self.num_samples = 4

	def __iter__(self):
		for i in range(0, 11, 2):
			yield i


class MySamplerTwo:
	def __init__(self):
		self.index_ids = [3, 4, 3, 2, 0, 11, 5, 5, 5, 9, 1, 11, 11, 11, 11, 8]

	def __getitem__(self, index):
		return self.index_ids[index]

	def __len__(self):
		return len(self.index_ids)
