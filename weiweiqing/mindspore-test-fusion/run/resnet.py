import mindspore as ms
from data import Cifar10Data, FoodData
from model import ResNet
from mindspore import nn, Tensor
from run import RunUtil, SimpleTrain
import numpy as np
import os
from matplotlib import pyplot as plt

os.environ['PYTHONHASHSEED'] = str(42)  # 为了禁止hash随机化，使得实验可复现
ms.set_seed(42)
RunUtil.ignore_warnings()
# RunUtil.set_graph_mode()

num_epochs = 200
batch_size = 128
num_classes = 92
workers = 4

data_dir = "/home/yang/fru92"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
resize = 256
crop = 224

train_dataset, val_dataset = FoodData.get_dataset(
	data_dir=data_dir,
	resize=resize,
	crop=crop,
	mean=mean,
	std=std,
	batch_size=batch_size,
	workers=workers
)
step_size_train = train_dataset.get_dataset_size()

# ms.set_seed(0)
# RunUtil.ignore_warnings()
# # RunUtil.set_graph_mode()
#
# image_size = 224
# batch_size = 128
# workers = 4
# num_classes = 92
# num_epochs = 200

# train_dataset, val_dataset = Cifar10Data.get_dataset(
# 	resize=image_size,
# 	batch_size=batch_size,
# 	workers=workers
# )
# step_size_train = train_dataset.get_dataset_size()

# visualization
# Cifar10Data.visualize(dataset_train)

net = ResNet.get_resnet152(num_classes=num_classes)

# print("\nResnet Model:")
# print(net)
#
# x = Tensor(np.random.randn(2, 3, 224, 224), ms.float32)
# print(net(x))
# exit()


lr = nn.cosine_decay_lr(
	min_lr=0.00001,
	max_lr=0.001,
	total_step=step_size_train * num_epochs,
	step_per_epoch=step_size_train,
	decay_epoch=num_epochs
)
optimizer = nn.Momentum(
	params=net.trainable_params(),
	learning_rate=lr,
	momentum=0.9
)

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

SimpleTrain(
	train_dataset=train_dataset,
	val_dataset=val_dataset,
	net=net,
	loss_fn=loss_fn,
	optimizer=optimizer,
	epochs=num_epochs
).run()


# visualize model
def visualize_model(best_ckpt_path, dataset_val):
	RunUtil.load_net(net, best_ckpt_path)

	data = next(dataset_val.create_dict_iterator())
	images = data["image"]
	labels = data["label"]

	output = net(images)
	pred = np.argmax(output.asnumpy(), axis=1)

	classes = []
	with open(os.path.join("../cifar-10-batches-bin", "batches.meta.txt"), "r") as f:
		for line in f:
			line = line.rstrip()
			if line:
				classes.append(line)

	plt.figure()
	for i in range(6):
		plt.subplot(2, 3, i + 1)
		color = "blue" if pred[i] == labels.asnumpy()[i] else "red"
		plt.title(f"Predict: { classes[pred[i]]}", color=color)
		picture_show = np.transpose(images.asnumpy()[i], (1, 2, 0))
		mean = np.array([0.4914, 0.4822, 0.4465])
		std = np.array([0.2023, 0.1994, 0.2010])
		picture_show = std * picture_show + mean
		picture_show = np.clip(picture_show, 0, 1)
		plt.imshow(picture_show)
		plt.axis('off')
	plt.show()


# visualize_model(os.path.join("./results", ""), val_dataset)
