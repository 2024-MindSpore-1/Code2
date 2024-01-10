# --------- from MindSpore Tutorial --------

from PIL import Image
from io import BytesIO
from mindspore.mindrecord import FileWriter
from mindspore.dataset import MindDataset
from mindspore.dataset.vision import Decode
import os

from .. import Cifar10Data


class DemoConvertor:
	def __init__(self, file_name):
		self.file_name = file_name

	def convert(self):
		cv_schema = {
			"file_name": {"type": "string"},
			"label": {"type": "int32"},
			"data": {"type": "bytes"}
		}

		# 声明MindSpore Record文件格式
		writer = FileWriter(self.file_name, shard_num=1, overwrite=True)
		writer.add_schema(cv_schema, "it is a cv dataset")
		writer.add_index(["file_name", "label"])

		# 创建数据集
		data = []
		for i in range(100):
			sample = {}
			white_io = BytesIO()
			Image.new('RGB', ((i+1)*10, (i+1)*10), (255, 255, 255)).save(white_io, 'JPEG')
			sample['file_name'] = str(i+1) + ".jpg"
			sample['label'] = i+1
			sample['data'] = white_io.getvalue()

			data.append(sample)
			if i % 10 == 0:
				writer.write_raw_data(data)
				data = []

		if data:
			writer.write_raw_data(data)

		writer.commit()

	def get_dataset(self):
		if not os.path.exists(self.file_name):
			self.convert()
		# 读取MindSpore Record文件格式
		data_set = MindDataset(dataset_files=self.file_name)
		decode_op = Decode()
		return data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)


class Cifar10Convertor:
	def __init__(self, save_path):
		self.save_path = save_path

	def convert(self):
		dataset = Cifar10Data.get_dataset_demo(self.save_path)
		dataset.save(os.path.join(self.save_path, "ifar10.mindrecord"))

	def get_dataset(self):
		path = os.path.join(self.save_path, "ifar10.mindrecord")
		dataset = MindDataset(dataset_files=path)

		# if os.path.exists("cifar10.mindrecord") and os.path.exists("cifar10.mindrecord.db"):
		# 	os.remove("cifar10.mindrecord")
		# 	os.remove("cifar10.mindrecord.db")

		return dataset


