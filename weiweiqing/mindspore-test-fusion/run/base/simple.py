from .. import RunUtil
import mindspore as ms
import time
from typing import Callable


class SimpleTrain:
	def __init__(
			self,
			train_dataset,
			val_dataset,
			net,
			loss_fn,
			optimizer,
			epochs,
			rel_path="./results",
			initialize_parameters=False,
			lr_scheduler: Callable = None,
	):
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.train_dataloader = train_dataset.create_tuple_iterator(num_epochs=epochs)
		if self.val_dataset is not None:
			self.val_dataloader = val_dataset.create_tuple_iterator(num_epochs=epochs)

		self.net = net
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.lr_scheduler: Callable = lr_scheduler
		self.epochs = epochs
		self.best_accuracy = 0.0
		self.best_epoch = 0
		self.rel_path = rel_path

		if initialize_parameters:
			RunUtil.init_parameters(self.net)

		# Get gradient function
		self.grad_fn = ms.value_and_grad(self.forward_fn, None, optimizer.parameters, has_aux=True)

	# define forward function
	def forward_fn(self, data, label):
		pred = self.net(data)
		loss = self.loss_fn(pred, label)
		return loss, pred

	# Define function of one-step training
	# @ms.jit
	def train_step(self, data, label):
		(loss, _), grads = self.grad_fn(data, label)
		self.optimizer(grads)
		return loss

	# define training function
	def train_epoch(self, epoch):
		num_batches = self.train_dataset.get_dataset_size()
		self.net.set_train()
		count = 0
		for data, label in self.train_dataloader:
			loss = self.train_step(data, label)
			if count % 100 == 0:
				loss, current = loss.asnumpy(), count
				print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}]")
			count += 1

	# define validation function
	def valid(self, epoch):
		num_batches = self.val_dataset.get_dataset_size()
		self.net.set_train(False)
		total, test_loss, top1, top5 = 0, 0, 0, 0
		for data, label in self.val_dataloader:
			pred = self.net(data)
			total += len(data)
			test_loss += self.loss_fn(pred, label).asnumpy()
			top1 += (pred.argmax(axis=1) == label).asnumpy().sum()
			_, indices = pred.topk(k=5, dim=1)
			top5 += (indices == label.unsqueeze(dim=0).t().expand_as(indices)).asnumpy().sum()
		test_loss /= num_batches
		top1 /= total
		top5 /= total
		print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Test: \n Top1: {(100*top1):>0.1f}%, Top5: {(100*top5):>0.1f}%, Avg loss: {test_loss:>8f} \n")
		if isinstance(self, SimpleTrain) and top1 > self.best_accuracy:
			RunUtil.remove_ckpt_file(net=self.net, path=self.rel_path, accuracy=self.best_accuracy, epoch=self.best_epoch)
			save_path = RunUtil.save_net(net=self.net, path=self.rel_path, accuracy=top1, epoch=epoch)
			print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Note: Checkpoint file has been saved to {save_path}")
			self.best_accuracy = top1
			self.best_epoch = epoch

	def run(self):
		self.best_accuracy = 0.0
		self.best_epoch = 0
		for epoch in range(self.epochs):
			print(f"\nEpoch: {epoch} \n----------------------------------")
			self.train_epoch(epoch)
			if self.val_dataset is not None:
				self.valid(epoch)
			if self.lr_scheduler:
				self.lr_scheduler(self.optimizer, epoch)
		print("\nDone!")


class SimpleEvaluation:
	def __init__(
			self,
			dataset,
			net,
			loss_fn,
			ckpt_path,
	):
		self.val_dataset = dataset
		self.net = net
		self.loss_fn = loss_fn
		self.ckpt_path = ckpt_path

	def run(self):
		RunUtil.load_net(self.net, self.ckpt_path)
		SimpleTrain.valid(self=self)
