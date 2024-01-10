import mindspore as ms
from mindspore import nn, Tensor, ops
from run import RunUtil, SimpleTrain, AMPTrain
from data.food_data import FoodData
from model.main_model import MODEL, SENet_BASE, MODEL_Another
import numpy as np
import os

os.environ['PYTHONHASHSEED'] = str(42)
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

learning_rate = 1e-2
init_weight_path = "resnet50_224_new.ckpt"

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

# test dataset
# image, label = next(train_dataset.create_tuple_iterator())
# print(f"Image Shape: {image.shape}, Image Type: {type(image)}, Label Shape: {label.shape}, Label Type: {type(label)}")
# exit()

net = MODEL_Another(
	num_classes=num_classes,
	resnet50_weight=init_weight_path,
	multi_scale=True,
)

# print(f"\nMODEL model:")
# print(net)

# test model
# x = Tensor(np.random.randn(2, 3, 224, 224), ms.float32)
# y = net(x)
# print(y.shape)
# exit()

loss_fn = nn.CrossEntropyLoss()

global_out_parameters = list(net.global_out.trainable_params())
global_out_parameter_ids = list(map(id, global_out_parameters))  # layer need to be trained
base_params = list(filter(lambda p: id(p) not in global_out_parameter_ids, net.trainable_params()))


optimizer = nn.SGD([
		{'params': base_params, 'lr': learning_rate},
		{'params': global_out_parameters, 'lr': learning_rate * 10}
	],
	learning_rate=learning_rate,
	momentum=0.9,
	weight_decay=0.00001
)

# region obsolete

cosine_decay_scheduler = nn.CosineDecayLR(
	min_lr=0.0,
	max_lr=learning_rate,
	decay_steps=num_epochs,
)


def lr_scheduler(optim, epoch):
	lr = cosine_decay_scheduler(Tensor(epoch, ms.int32))
	for param in optim.group_lr[0:len(base_params)]:
		param.set_data(lr)
	for param in optim.group_lr[len(base_params):]:
		param.set_data(lr * 10)


# test learning rate scheduler
# for i in range(1, epochs):
# 	lr_scheduler(optimizer, i)
# 	print(list(map(lambda param: param.value().asnumpy().item(), optimizer.group_lr[0:len(base_params)])))
# 	print(list(map(lambda param: param.value().asnumpy().item(), optimizer.group_lr[len(base_params):])))
# exit()

# endregion

# AMPTrain(
# 	train_dataset=train_dataset,
# 	val_dataset=val_dataset,
# 	net=net,
# 	loss_fn=loss_fn,
# 	optimizer=optimizer,
# 	lr_scheduler=lr_scheduler,
# 	epochs=num_epochs,
# ).run()


SimpleTrain(
	train_dataset=train_dataset,
	val_dataset=val_dataset,
	net=net,
	loss_fn=loss_fn,
	optimizer=optimizer,
	lr_scheduler=None,
	epochs=num_epochs,
).run()
