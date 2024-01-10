from data import ImageNetData
from loss import CrossEntropySmooth
from model import ViT
from run import SimpleTrain, AMPTrain, AMPGpuTrain, RunUtil
import numpy as np
from mindspore import Tensor, dtype, nn
import mindspore as ms
from download import download

ms.set_seed(0)
RunUtil.ignore_warnings()
# RunUtil.set_graph_mode()

num_classes = 1000
batch_size = 16
num_epochs = 10

train_dataset, val_dataset = ImageNetData.get_dataset(batch_size=batch_size)
step_size = train_dataset.get_dataset_size()

net = ViT(num_classes=num_classes)

print("/nVision Transformer Model: ")
print(net)

# x = Tensor(np.random.randn(2, 3, 224, 224), dtype=dtype.float32)
# y = net(x)
# print(y.shape)
# exit()

# load check point
vit_url = "https://download.mindspore.cn/vision/classification/vit_b_16_224.ckpt"
path = "./ckpt/vit_b_16_224.ckpt"

vit_path = download(vit_url, path, replace=True)
RunUtil.load_net(net, path)

# define learning rate
lr = nn.cosine_decay_lr(
	min_lr=float(0),
	max_lr=0.00005,
	total_step=num_epochs * step_size,
	step_per_epoch=step_size,
	decay_epoch=10
)

# define optimizer
optimizer = nn.Adam(net.trainable_params(), lr)

# define loss function
loss_fn = CrossEntropySmooth(
	sparse=True,
	reduction="mean",
	smooth_factor=0.1,
	num_classes=num_classes
)

RunUtil.get_train_engine()(
	train_dataset=train_dataset,
	val_dataset=val_dataset,
	net=net,
	loss_fn=loss_fn,
	optimizer=optimizer,
	epochs=num_epochs
).run()
