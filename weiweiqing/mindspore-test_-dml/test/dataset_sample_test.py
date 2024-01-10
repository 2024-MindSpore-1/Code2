from data import Cifar10Data
import numpy as np
from matplotlib import pyplot as plt
import math

from mindspore.dataset import NumpySlicesDataset, RandomSampler,\
	WeightedRandomSampler, SubsetSampler, PKSampler, DistributedSampler


def plt_rel(dataset, rows):
	# cols = math.ceil(dataset.get_dataset_size() / rows)
	print(dataset.get_dataset_size())
	num = 0
	for data in dataset.create_dict_iterator(output_numpy=True):
		print("Image Shape: ", data["image"].shape, "Label:", data["label"])
		# plt.subplot(rows, cols, num)
		# plt.imshow(data["image"], interpolation=None)
		num += 1
	print(f"num = {num}")
	# plt.show()


# region RandomSampler

# np_data = [1, 2, 3, 4, 5, 6, 7, 8]
# sampler1 = RandomSampler(replacement=True, num_samples=5)
# dataset1 = NumpySlicesDataset(np_data, column_names=["data"], sampler=sampler1)
#
# print("With replacement: ", end='')
# for data in dataset1.create_tuple_iterator():
# 	print(data[0], end=" ")
#
# sampler2 = RandomSampler(replacement=False, num_samples=5)
# dataset2 = NumpySlicesDataset(np_data, column_names=["data"], sampler=sampler2)
# print("\nWithout replacement: ", end='')
# for data in dataset2.create_tuple_iterator():
# 	print(data[0], end=" ")

# endregion


# region WeightedRandomSampler

# weights = [0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# sampler = WeightedRandomSampler(weights=weights, num_samples=6)

# endregion

# region SubsetRandomSampler
#
# indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# sampler = SubsetSampler(indices=indices, num_samples=6)

# endregion

# region PKSampler

# # 每种类别抽样2个样本，最多10个样本
# sampler = PKSampler(num_val=2, class_column="label", num_samples=60, shuffle=True)
#
# # endregion
#
# dataset = Cifar10Data.get_dataset_demo(sampler=sampler, batch_size=8)
# plt_rel(dataset, 2)


# region DistributedSampler

# # 自定义数据集
# data_source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#
# # 构建的数据集分为4片，共采样3个数据样本
# sampler = DistributedSampler(num_shards=4, shard_id=0, shuffle=False, num_samples=3)
# dataset = NumpySlicesDataset(data_source, column_names=["data"], sampler=sampler)
#
# # 打印数据集
# for data in dataset.create_dict_iterator():
# 	print(data)

# endregion


from data.sampler.demo import MySampler, MySamplerTwo

# 自定义数据集
np_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

# 加载数据
dataset = NumpySlicesDataset(np_data, column_names=["data"], sampler=MySampler())
dataset = dataset.batch(2)
for data in dataset.create_tuple_iterator(output_numpy=True):
	print(data[0])
