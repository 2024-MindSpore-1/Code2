# ------- from MindSpore Tutorial ----------

import mindspore.dataset as ds


class MySampler(ds.Sampler):
	def __iter__(self):
		for i in range(0, 10, 2):
			yield i


class MySamplerTwo:
	def __init__(self):
		self.index_ids = [3, 4, 3, 2, 0, 11, 5, 5, 5, 9, 1, 11, 11, 11, 11, 8]

	def __getitem__(self, index):
		return self.index_ids[index]

	def __len__(self):
		return len(self.index_ids)
