import os
from run.run_dml import opt
import mindspore as ms
from mindspore import nn, Tensor
from run import RunUtil

import architectures
import model
import numpy as np
from mindspore.common.initializer import Normal

from data.dataset_dml import get_datasets

from loss.criteria import get_criterion
from loss.batch_miner import get_batch_miner
from metric.metric_dml.metric_computer import MetricComputer
from architectures import get_arch
from data.sampler_dml import get_data_sampler

from util.util_dml import logger, misc
from run.run_dml.dml_simple_train import DmlSimpleTrain

ms.set_seed(opt.seed)
RunUtil.ignore_warnings()

# region test Vit model

# net = model.vit.ViT(num_classes=1000)
# params_dict = ms.load_checkpoint("vit_b_16_224.ckpt")
# ms.load_param_into_net(net, params_dict)
#
# x = Tensor(shape=(2, 3, 224, 224), dtype=ms.float32, init=Normal())
#
# y = net(x)
# print(y.shape)
#
# exit()

# endregion

# region test Resnet50 model

# net = model.resnet.ResNet.get_resnet50(num_classes=1000)
# params_dict = ms.load_checkpoint("resnet50_224_new.ckpt")
# ms.load_param_into_net(net, params_dict)
#
# x = Tensor(shape=(2, 3, 224, 224), dtype=ms.float32, init=Normal())
#
# y = net(x)
# print(y.shape)
#
# exit()

# endregion

# region test vit architecture model

# net = architectures.vit.Vit(opt, "vit_b_16_224.ckpt")
#
# x = Tensor(shape=(2, 3, 224, 224), dtype=ms.float32, init=Normal())
#
# y, z = net(x)
# print(y.shape, z.shape)
#
# exit()

# endregion

# region test resnet50 architecture model

# net = architectures.resnet50.Resnet50(opt, "resnet50_224_new.ckpt")
#
# x = Tensor(shape=(2, 3, 224, 224), dtype=ms.float32, init=Normal())
#
# y, (z, w) = net(x)
# print(y.shape, z.shape, w.shape)
#
# exit()

# endregion


# region wandb

if opt.save_name == "group_plus_seed":
	if opt.log_online:
		opt.save_name = f"{opt.group}_s{opt.seed}"
	else:
		opt.save_name = ""

if opt.log_online:
	import wandb
	_ = os.system('wandb login {}'.format(opt.wandb_key))
	os.environ['WANDB_API_KEY'] = opt.wandb_key
	wandb.init(project=opt.project, group=opt.group, name=opt.save_name, dir=opt.save_path)
	wandb.config.update(opt)

# endregion

opt.save_path += "/" + opt.dataset  # result path

# check sampler_per_class and batch size
assert not opt.bs % opt.samples_per_class, 'Batch size needs to fit number of samples per class for distance sampling and margin/triplet loss!'

# whether load pretrained parameters
opt.pretrained = not opt.not_pretrained

# GPU settings, now only one gpu used
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])


# dataloader/datasets setups

# train_data_sampler = None
dataloaders = {}
datasets = get_datasets(opt.dataset, opt, opt.source_path)

dataloaders["evaluation"] = datasets["evaluation"].generate_dataset()
dataloaders["testing"] = datasets["testing"].generate_dataset()
if opt.use_tv_split:
	dataloaders["validation"] = datasets["validation"].generate_dataset()

data_sampler = get_data_sampler(opt.data_sampler, opt, datasets["training"].image_dict, datasets["training"].image_list)

dataloaders["training"] = datasets["training"].generate_dataset(data_sampler=data_sampler)

opt.n_classes = len(datasets["training"].avail_classes)

# ----- network setup !!!!!!
model = get_arch(opt.arch, opt=opt, weight_file_path="vit_b_16_224.ckpt")
# model = get_arch(opt.arch, opt=opt, weight_file_path="resnet50_224_new.ckpt")


# set last linear layer's learning rate
if opt.fc_lr < 0:
	to_optim = [{'params': model.trainable_params(), 'lr': opt.lr, 'weight_decay': opt.decay}]
else:
	all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'last_linear' not in x[0], model.parameters_and_names()))]
	fc_params = model.last_linear.parameters()
	to_optim = [{'params': all_but_fc_params, 'lr': opt.lr, 'weight_decay': opt.decay},
				{'params': fc_params, 'lr': opt.fc_lr, 'weight_decay': opt.decay}]


# create logging files
sub_loggers = ["Train", "Test", "Model Grad"]
if opt.use_tv_split:
	sub_loggers.append('Val')
LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)


# loss setup
batch_miner = get_batch_miner(opt.batch_mining, opt)
criterion, to_optim = get_criterion(opt.loss, opt, to_optim, batch_miner)
criterion_m, to_optim = get_criterion(opt.m_loss, opt, to_optim, batch_miner)
#
# if 'criterion' in train_data_sampler.name:
# 	train_data_sampler.internal_criterion = criterion


# optim setup
scheduler = nn.piecewise_constant_lr(milestone=opt.tau, learning_rates=opt.gamma)
if opt.optim == "adam":
	optimizer = nn.Adam(to_optim, learning_rate=scheduler)
elif opt.optim == "sgd":
	optimizer = nn.SGD(to_optim, momentum=0.9, learning_rate=scheduler)
else:
	raise ValueError(f"Optimizer <{opt.optim}> is not available!")


# metric computer
opt.rho_spectrum_embed_dim = opt.embed_dim
metric_computer = MetricComputer(opt.evaluation_metrics, opt)


# Print Summary
data_text = f"Dataset:\t{opt.dataset.upper()}"
setup_text = f"Objective:\t{opt.loss.upper()}"
miner_text = f"Batch Miner:\t{opt.batch_mining if criterion.REQUIRES_BATCHMINER else 'N/A' }"
arch_text = f"Backbone:\t{opt.arch.upper()} (weights: {misc.gimme_params(model)})"
summary = "\n".join([data_text, setup_text, miner_text, arch_text])
print(summary)


# === Train & Evaluate ===
DmlSimpleTrain(
	opt=opt,
	train_dataset=dataloaders["training"],
	val_dataset=dataloaders["validation"] if opt.use_tv_split else None,
	test_dataset=dataloaders["testing"],
	eval_dataset=dataloaders["evaluation"],
	net=model,
	loss_fn=criterion,
	optimizer=optimizer,
	metric_computer=metric_computer,
	LOG=LOG,
	loss_fn_m=criterion_m
).run()



