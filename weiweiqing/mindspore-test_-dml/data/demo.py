import os

from download import download
import mindspore as ms
from mindspore.dataset import vision, transforms, GeneratorDataset
from mindspore.dataset import MnistDataset, Cifar10Dataset, ImageFolderDataset
from matplotlib import pyplot as plt
import numpy as np


class MnistData:
	@staticmethod
	def get_dataset(save_path="../", batch_size=64, shuffle=True):
		url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
		path = download(url=url, path=save_path, kind="zip", replace=False)

		train_dataset = MnistDataset(save_path + "MNIST_Data/train")
		val_dataset = MnistDataset(save_path + "MNIST_Data/test")

		print(f"\nMnist Dataset loaded, column names: ")
		print(train_dataset.get_col_names())

		image_transforms = [
			vision.Rescale(1.0 / 255.0, 0),
			vision.Normalize(mean=(0.1307,), std=(0.3081,)),
			vision.HWC2CHW()
		]

		label_transform = transforms.TypeCast(ms.int32)

		train_dataset = train_dataset.map(image_transforms, "image").map(label_transform, "label").batch(batch_size)
		if shuffle:
			train_dataset = train_dataset.shuffle(buffer_size=64)
		val_dataset = val_dataset.map(image_transforms, "image").map(label_transform, "label").batch(batch_size)

		print("\nTuple Iteration Mode:")
		for image, label in train_dataset.create_tuple_iterator():
			print(f"Shape of image [N C H W]: {image.shape}  {image.dtype}")
			print(f"Shape of label: {label.shape}  {label.dtype}")
			break

		print("\nDict Iteration Mode:")
		for item in train_dataset.create_dict_iterator():
			print(f"Shape of image [N C H W]: {item['image'].shape}  {item['image'].dtype}")
			print(f"Shape of label: {item['label'].shape}  {item['label'].dtype}")
			break

		return train_dataset, val_dataset

	@staticmethod
	def visualize(save_path="../"):
		url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
		path = download(url=url, path=save_path, kind="zip", replace=False)

		train_dataset = MnistDataset(save_path + "MNIST_Data/train", shuffle=False)

		figure = plt.figure(figsize=(4, 4))
		cols, rows = 3, 3

		for idx, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
			figure.add_subplot(rows, cols, idx + 1)
			plt.title(int(label))
			plt.axis("off")
			plt.imshow(image.asnumpy().squeeze(), cmap="gray")
			if idx == cols * rows - 1:
				break

		plt.show()


class Cifar10Data:
	@staticmethod
	def get_dataset_demo(sampler, save_path="../", batch_size=None):
		url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
		path = download(url, save_path, kind="tar.gz", replace=False)

		data_dir = os.path.join(save_path, "cifar-10-batches-bin/")
		dataset = Cifar10Dataset(data_dir, sampler=sampler)
		print(dataset.get_dataset_size())
		if batch_size is not None:
			dataset = dataset.batch(batch_size)
		return dataset

	@staticmethod
	def _get_dataset(
			usage,
			resize,
			batch_size=256,
			workers=4,
			save_path="../"
	):
		url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
		path = download(url, save_path, kind="tar.gz", replace=True)

		data_dir = os.path.join(save_path, "cifar-10-batches-bin")

		dataset = Cifar10Dataset(
			dataset_dir=data_dir,
			usage=usage,
			num_parallel_workers=workers,
		)

		trans = []
		if usage == "train":
			dataset = dataset.shuffle(buffer_size=64)
			trans += [
				# vision.RandomCrop((32, 32), (4, 4, 4, 4)),
				vision.RandomHorizontalFlip(prob=0.5)
			]

		trans += [
			vision.Resize(resize),
			vision.Rescale(1.0 / 255.0, 0.0),
			vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
			vision.HWC2CHW()
		]

		target_trans = transforms.TypeCast(ms.int32)

		dataset = dataset.map(
			operations=trans,
			input_columns="image",
			num_parallel_workers=workers,
		)
		dataset = dataset.map(
			operations=target_trans,
			input_columns="label",
			num_parallel_workers=workers,
		)

		dataset = dataset.batch(batch_size)

		return dataset

	@staticmethod
	def get_dataset(
			resize,
			batch_size=256,
			workers=4,
			save_path="../"
	):
		train_dataset = Cifar10Data._get_dataset(
			usage="train",
			resize=resize,
			batch_size=batch_size,
			workers=workers,
			save_path=save_path
		)
		val_dataset = Cifar10Data._get_dataset(
			usage="test",
			resize=resize,
			batch_size=batch_size,
			workers=workers,
			save_path=save_path
		)

		return train_dataset, val_dataset

	@staticmethod
	def visualize(
			dataset,
			save_path="../"
	):
		data_dir = os.path.join(save_path, "cifar-10-batches-bin")
		classes = []
		with open(os.path.join(data_dir, "batches.meta.txt"), "r") as f:
			for line in f:
				line = line.rstrip()
				if line:
					classes.append(line)

		# get images of first batch
		data = next(dataset.create_dict_iterator())
		images = data["image"].asnumpy()
		labels = data["label"].asnumpy()
		print(f"Images shape: {images.shape}, Labels Shape: {labels.shape}")

		# first six pictures
		print(f"Labels: {labels[:6]}")

		plt.figure()
		for i in range(6):
			plt.subplot(2, 3, i + 1)
			plt.title(f"{classes[labels[i]]}")
			plt.axis("off")

			# recover image
			img = np.transpose(images[i], (1, 2, 0))
			mean = np.array([0.4914, 0.4822, 0.4465])
			std = np.array([0.2023, 0.1994, 0.2010])
			img = img * std + mean
			img = np.clip(img, 0, 1)

			plt.imshow(img)
		plt.show()


class ImageNetData:
	@staticmethod
	def _get_dataset(
			usage,
			batch_size=16,
			save_path="../"
	):
		dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/vit_imagenet_dataset.zip"
		path = download(dataset_url, save_path, kind="zip", replace=False)

		data_path = "../dataset/"
		mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
		std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

		if usage == "train":
			dataset = ImageFolderDataset(os.path.join(data_path, "train"), shuffle=True)
			trans = [
				vision.RandomCropDecodeResize(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
				vision.RandomHorizontalFlip(prob=0.5),
				vision.Normalize(meant=mean, std=std),
				vision.HWC2CHW()
			]
		else:
			dataset = ImageFolderDataset(os.path.join(data_path, "val"), shuffle=True)
			trans = [
				vision.Decode(),
				vision.Resize(224 + 32),
				vision.CenterCrop(224),
				vision.Normalize(mean=mean, std=std),
				vision.HWC2CHW()
			]

		dataset = dataset.map(operations=trans, input_columns=["image"])
		dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

		return dataset

	@staticmethod
	def get_dataset(batch_size=16, save_path="../"):
		train_dataset = ImageNetData._get_dataset(usage="train", batch_size=batch_size, save_path=save_path)
		val_dataset = ImageNetData._get_dataset(usage="val", batch_size=batch_size, save_path=save_path)
		return train_dataset, val_dataset

	@staticmethod
	def get_train_dataset(batch_size=16, save_path="../"):
		return ImageNetData._get_dataset(usage="train", batch_size=batch_size, save_path=save_path)

	@staticmethod
	def get_val_dataset(batch_size=16, save_path="../"):
		return ImageNetData._get_dataset(usage="val", batch_size=batch_size, save_path=save_path)


class Iterable:
	def __init__(self):
		self._data = np.random.sample((5, 2))
		self._label = np.random.sample((5, 1))

	def __getitem__(self, index):
		return self._data[index], self._label[index]

	def __len__(self):
		return len(self._data)

	@staticmethod
	def get_dataset():
		return GeneratorDataset(source=Iterable(), column_names=["data", "label"], shuffle=False, num_parallel_workers=1)


class Iterator:
	def __init__(self):
		self._index = -1
		self._data = ms.Tensor(np.random.sample((5, 2)), ms.float32)
		self._label = ms.Tensor(np.random.sample((5, 1)), ms.float32)

	def __next__(self):
		self._index += 1
		if self._index >= len(self._data):
			raise StopIteration()
		else:
			return self._data[self._index, :], self._label[self._index, :]

	def __iter__(self):
		self._index = -1
		return self

	@staticmethod
	def get_dataset():
		return GeneratorDataset(source=Iterator(), column_names=["data", "label"], shuffle=False, num_parallel_workers=1)
