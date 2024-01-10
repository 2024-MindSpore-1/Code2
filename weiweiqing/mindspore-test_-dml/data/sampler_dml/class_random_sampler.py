import random
from . import register_data_sampler


@register_data_sampler
class ClassRandomSampler:
	def __init__(self, opt, image_dict, image_list):
		super().__init__()
		self.image_dict = image_dict
		self.image_list = image_list
		self.image_count = len(image_list)

		#####
		self.classes = list(self.image_dict.keys())

		####
		self.batch_size = opt.bs
		self.samples_per_class = opt.samples_per_class
		self.sampler_length = len(image_list) // opt.bs
		assert self.batch_size % self.samples_per_class == 0, '#Samples per class must divide batch size!'
		# self.draws = self.batch_size // self.samples_per_class

		self.name = 'class_random_sampler'
		self.requires_storage = False

		self.eleIndex_in_cls = 0
		self.clsCur = None
		self.iter_len = self.sampler_length * self.batch_size

	def __getitem__(self, index):

		if index == self.iter_len:
			raise StopIteration()

		self.eleIndex_in_cls = index % self.samples_per_class

		# select a class randomly
		if self.eleIndex_in_cls == 0:
			self.clsCur = random.choice(self.classes)

		return random.choice(self.image_dict[self.clsCur])[-1]

	def __len__(self):
		# return len(self.sampler_length)
		return len(self.image_list)
