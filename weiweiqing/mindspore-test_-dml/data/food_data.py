from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision, transforms
import mindspore as ms
import os
from PIL import Image


class FoodData:
	def __init__(
			self,
			data_dir,
			usage,
	):
		self.data_dir = data_dir
		self.img_entries = []
		data_txt_file = os.path.join(
			data_dir,
			f"{usage}.txt"
		)
		with open(data_txt_file, "r") as f:
			for line in f:
				line = line.strip()
				if line:
					words = line.split(" ")
					self.img_entries.append((" ".join(words[:-1]), int(words[-1]) - 1))

	def __getitem__(self, index):
		image_file_path, label = self.img_entries[index]
		image = Image.open(os.path.join(self.data_dir, "images", image_file_path)).convert("RGB")
		return image, label

	def __len__(self):
		return len(self.img_entries)

	@staticmethod
	def get_dataset(
			data_dir,
			resize,
			crop,
			mean,
			std,
			batch_size,
			workers,
	):
		train_trans = [
				vision.RandomHorizontalFlip(prob=0.5),
				vision.Resize(resize),
				vision.RandomCrop(crop),
				vision.Rescale(1.0 / 255.0, 0),
				vision.Normalize(mean=mean, std=std),
				vision.HWC2CHW()
			]

		val_trans = [
				vision.Resize(resize),
				vision.CenterCrop(crop),
				vision.Rescale(1.0 / 255.0, 0),
				vision.Normalize(mean=mean, std=std),
				vision.HWC2CHW()
			]

		label_tran = transforms.TypeCast(ms.int32)

		train_dataset = GeneratorDataset(
			source=FoodData(
				usage="train",
				data_dir=data_dir,
			),
			column_names=["image", "label"],
			shuffle=True,
			num_parallel_workers=workers,
		)
		train_dataset = train_dataset.map(operations=train_trans, input_columns=["image"])
		train_dataset = train_dataset.map(operations=label_tran, input_columns=["label"])
		train_dataset = train_dataset.batch(batch_size)

		val_dataset = GeneratorDataset(
			source=FoodData(
				usage="val",
				data_dir=data_dir,
			),
			column_names=["image", "label"],
			shuffle=False,
			num_parallel_workers=workers,
		)
		val_dataset = val_dataset.map(operations=val_trans, input_columns=["image"])
		val_dataset = val_dataset.map(operations=label_tran, input_columns=["label"])
		val_dataset = val_dataset.batch(batch_size)

		return train_dataset, val_dataset

