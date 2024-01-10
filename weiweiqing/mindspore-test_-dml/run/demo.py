import mindspore as ms
from mindspore import nn
from data import MnistData
from model import DemoNetwork
from run import RunUtil, AMPTrain

ms.set_seed(0)
RunUtil.ignore_warnings()
# RunUtil.set_graph_mode()

train_dataset, val_dataset = MnistData.get_dataset()

net = DemoNetwork()
print("\nDemo Model:")
print(net)

# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(params=net.trainable_params(), learning_rate=1e-2)


AMPTrain(
	train_dataset=train_dataset,
	val_dataset=val_dataset,
	net=net,
	loss_fn=loss_fn,
	optimizer=optimizer,
	lr_scheduler=None,
	epochs=3
).run()

# SimpleEvaluation(
# 	dataset=val_dataset,
# 	net=net,
# 	loss_fn=loss_fn,
# 	ckpt_path="./results/DemoNetwork_0.92.ckpt"
# ).run()
