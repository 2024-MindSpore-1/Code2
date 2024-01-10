from ..base.simple import SimpleTrain
import time
from tqdm import tqdm
import numpy as np
from .dml_evaluation import evaluate
from mindspore import ops


class DmlSimpleTrain(SimpleTrain):
	def __init__(
			self,
			opt,
			train_dataset,
			val_dataset,
			test_dataset,
			eval_dataset,
			net,
			loss_fn,
			optimizer,
			metric_computer,
			LOG,
			loss_fn_m,
	):
		super().__init__(
			train_dataset,
			val_dataset,
			net,
			loss_fn,
			optimizer,
			opt.n_epochs,
		)
		self.opt = opt
		self.LOG = LOG
		self.metric_computer = metric_computer
		self.test_dataset = test_dataset
		self.eval_dataset = eval_dataset
		if test_dataset:
			self.test_dataloader = test_dataset.create_tuple_iterator(num_epochs=opt.n_epochs)
		if eval_dataset:
			self.eval_dataloader = eval_dataset.create_tuple_iterator(num_epochs=opt.n_epochs)

		self.loss_fn_m = loss_fn_m

	# ------ note: modify here !!! -------
	def forward_fn(self, input, class_labels):
		# compute embedding
		model_args = {'x': input}
		if 'mix' in self.opt.arch:
			model_args['labels'] = class_labels

		if self.opt.multi_loss:
			embeds = self.net(**model_args)
			embeds_cls, embeds_patch = embeds

			loss_args_cls = {}
			loss_args_patch = {}

			# Compute Loss
			loss_args_cls['batch'] = embeds_cls
			loss_args_patch['batch'] = embeds_patch
			loss_args_cls['labels'] = class_labels
			loss_args_patch['labels'] = class_labels
			loss_cls = self.loss_fn(**loss_args_cls)
			loss_patch = self.loss_fn_m(**loss_args_patch)

			loss = 0.1 * loss_cls + 0.9 * loss_patch
		else:
			embeds = self.net(**model_args)
			if isinstance(embeds, tuple):
				embeds, (avg_features, features) = embeds
			loss_args = {}
			loss_args['batch'] = embeds
			loss_args['labels'] = class_labels
			loss_args['f_embed'] = self.net.resnet_model.fc
			loss_args['batch_features'] = features

			loss = self.loss_fn(**loss_args)

		return loss, embeds

	def train_step(self, input, class_labels):
		(loss, _), grads = self.grad_fn(input, class_labels)
		self.optimizer(grads)
		return loss, grads

	def train_epoch(self, epoch):
		start = time.time()
		self.net.set_train()

		loss_collect = []

		data_iterator = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
		for i, out in enumerate(data_iterator):
			# print(f"\nEpoch {epoch}, iteration {i} start...\n")

			class_labels, input, input_indices = out  # get input
			loss, grads = self.train_step(input, class_labels)  # train one epoch

			# compute model gradients and log them
			# grads = grads.asnumpy().flatten()
			# grads = ops.stack((param.flatten() for param in grads)).asnumpy()
			# grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
			# self.LOG.progress_saver['Model Grad'].log('Grad L2',  grad_l2,  group='L2')
			# self.LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

			# print(f"\nEpoch {epoch}, iteration {i} completed!\n")

			# record loss list
			loss_collect.append(float(loss))

			if i == len(self.train_dataset) - 1:
				data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))

		result_metrics = {'loss': np.mean(loss_collect)}

		self.LOG.progress_saver['Train'].log('epochs', epoch)
		for metricname, metricval in result_metrics.items():
			self.LOG.progress_saver['Train'].log(metricname, metricval)
		self.LOG.progress_saver['Train'].log('time', np.round(time.time()-start, 4))

	def valid_epoch(self):
		self.net.set_train(False)
		print("\nComputer Testing Metrics...")
		evaluate(self.LOG, self.metric_computer, self.test_dataloader, self.test_dataset, self.net, self.opt, self.opt.evaltypes, log_key="Test")
		# if self.opt.use_tv_split:
		# 	print("\nComputer Validation Metrics...")
		# 	evaluate(self.LOG, self.metric_computer, self.val_dataloader, self.val_dataset, self.net, self.opt, self.opt.evaltypes, log_key="Val")
		# print("\nComputer Training Metrics...")
		# evaluate(self.LOG, self.metric_computer, self.eval_dataloader, self.eval_dataset, self.net, self.opt, self.opt.evaltypes, log_key="Train")

	def run(self):
		print("\n-----\n")
		full_training_start_time = time.time()
		for epoch in range(self.opt.n_epochs):
			epoch_start_time = time.time()
			self.train_epoch(epoch)
			self.valid_epoch()
			self.LOG.update(all=True)
			print("Total Epoch Runtime: {0:4.2f}s".format(time.time()-epoch_start_time))

		# create a summary text file
		summary_text = ""
		full_training_time = time.time() - full_training_start_time
		summary_text += "Training Time: {0:.2} min.".format(full_training_time / 60)

		summary_text += "-----------------------\n"
		for sub_logger in self.LOG.sub_loggers:
			metrics = self.LOG.graph_writer[sub_logger].ov_title
			summary_text += "{} metrics: {}".format(sub_logger.upper(), metrics)

		with open(self.opt.save_path + "/training_summary.txt", 'w') as summary_file:
			summary_file.write(summary_text)
