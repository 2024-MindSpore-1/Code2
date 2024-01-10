import mindspore as ms
from mindspore import nn, Tensor, ops
from run import RunUtil, SimpleTrain, AMPTrain, AMPGpuTrain
from data.food_data import FoodData
from model.main_model import MODEL, SENet_BASE
import numpy as np
import os
from model.senet import senet154

os.environ['PYTHONHASHSEED'] = str(1029)  # 为了禁止hash随机化，使得实验可复现
ms.set_seed(42)
RunUtil.ignore_warnings()
RunUtil.set_graph_mode()

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

net = senet154(num_classes=num_classes, pretrained=None)

# print(f"\nMODEL model:")
# print(net)

# test model
# x = Tensor(np.random.randn(2, 3, 224, 224), ms.float32)
# y = net(x)
# print(y.shape)
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

loss_fn = nn.CrossEntropyLoss()


AMPTrain(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    net=net,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=num_epochs,
).run()



