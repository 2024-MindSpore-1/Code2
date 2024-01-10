import mindspore as ms
from mindspore import nn, Tensor
from run import RunUtil
from run.run_fusion import args
from model.model_fusion.resnet import ResNet, resnet50, resnet101
from model.model_fusion.resnet_concat import Resnet101_concat
from data.dataset_fusion.nutrition_rgbd import Nutrition_RGBD
from run.run_fusion.fusion_simple_train import FusionSimpleTrain
from mindspore.common.initializer import Normal


ms.set_seed(args.seed)
RunUtil.ignore_warnings()

print('==> Building model..')

net = resnet101(rgbd=args.rgbd)
net2 = resnet101(rgbd=args.rgbd)
net_cat = Resnet101_concat()

RunUtil.load_net(net, "food2k_resnet101_0.0001.ckpt")
RunUtil.load_net(net2, "food2k_resnet101_0.0001.ckpt")

# region test model

# inputs = Tensor(shape=(2, 3, 320, 448), dtype=ms.float32, init=Normal())
# inputs_rgbd = Tensor(shape=(2, 3, 320, 448), dtype=ms.float32, init=Normal())
#
# outputs = net(inputs)
# p2, p3, p4, p5 = outputs
# outputs_rgbd = net2(inputs_rgbd)
# d2, d3, d4, d5 = outputs_rgbd
# outputs = net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])
#
# print(len(outputs))
# exit(0)

# endregion

# Data
print("==> Preparing Data...")

train_dataset, test_dataset = Nutrition_RGBD.get_dataset(args=args)

# test dataset
# image_rgb, label, total_calories, total_mass, total_fat, total_carb, total_protein, img_rgbd = next(train_dataset.create_tuple_iterator())
# print(image_rgb.shape, img_rgbd.shape)
# print(total_fat)  #, total_carb, total_protein, total_mass, total_calories)
# exit()

# print(f'learning rate:{args.lr}, weight decay: {args.wd}')

criterion = nn.L1Loss()

num_patches = train_dataset.get_dataset_size()
scheduler = nn.ExponentialDecayLR(
	learning_rate=args.lr,
	decay_steps=num_patches,
	decay_rate=0.99,
)

net2_params = net2.trainable_params()
for para in net2_params:
	para.name = "net2." + para.name

net_cat_params = net_cat.trainable_params()
for para in net_cat_params:
	para.name = "net_cat." + para.name

params_all = net.trainable_params() + net2_params + net_cat_params
optimizer = nn.Adam(params_all, learning_rate=scheduler, weight_decay=5e-4)


# optimizer = nn.Adam([
# 	{'params': net.trainable_params(), 'lr': scheduler, 'weight_decay': 5e-4},  # lr: 5e-5
# 	{'params': net2.trainable_params(), 'lr': scheduler, 'weight_decay': 5e-4},
# 	{'params': net_cat.trainable_params(), 'lr': scheduler, 'weight_decay': 5e-4}
# ])

# exit(0)

FusionSimpleTrain(
	train_dataset=train_dataset,
	val_dataset=test_dataset,
	net=net,
	loss_fn=criterion,
	optimizer=optimizer,
	epochs=args.epochs,
	net2=net2,
	net_cat=net_cat,
).run()

