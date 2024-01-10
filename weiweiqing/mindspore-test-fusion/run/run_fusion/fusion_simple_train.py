from run.base.simple import SimpleTrain
from tqdm import tqdm


class FusionSimpleTrain(SimpleTrain):
	def __init__(
			self,
			train_dataset,
			val_dataset,
			net,
			loss_fn,
			optimizer,
			epochs,
			net2,
			net_cat,
	):
		super().__init__(
			train_dataset=train_dataset,
			val_dataset=val_dataset,
			net=net,
			loss_fn=loss_fn,
			optimizer=optimizer,
			epochs=epochs,
		)

		self.net2 = net2
		self.net_cat = net_cat

	def forward_fn(self, x):
		inputs = x[0]
		total_calories = x[2]
		total_mass = x[3]
		total_fat = x[4]
		total_carb = x[5]
		total_protein = x[6]
		inputs_rgbd = x[7]

		calories_per = total_calories/total_mass
		fat_per = total_fat/total_mass
		carb_per = total_carb/total_mass
		protein_per = total_protein/total_mass

		outputs = self.net(inputs)
		p2, p3, p4, p5 = outputs
		outputs_rgbd = self.net2(inputs_rgbd)
		d2, d3, d4, d5 = outputs_rgbd
		outputs = self.net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])

		# calories_per_loss = self.loss_fn(outputs[0], calories_per)
		# fat_per_loss = self.loss_fn(outputs[2], fat_per)
		# carb_per_loss = self.loss_fn(outputs[3], carb_per)
		# protein_per_loss = self.loss_fn(outputs[4], protein_per)

		# loss = calories_per_loss + fat_per_loss + carb_per_loss + protein_per_loss
		# return loss, calories_per_loss, fat_per_loss, carb_per_loss, protein_per_loss

		total_calories_loss = total_calories.shape[0] * self.loss_fn(outputs[0], total_calories) / total_calories.sum().item()
		total_mass_loss = total_calories.shape[0] * self.loss_fn(outputs[1], total_mass) / total_mass.sum().item()
		total_fat_loss = total_calories.shape[0] * self.loss_fn(outputs[2], total_fat) / total_fat.sum().item()
		total_carb_loss = total_calories.shape[0] * self.loss_fn(outputs[3], total_carb) / total_carb.sum().item()
		total_protein_loss = total_calories.shape[0] * self.loss_fn(outputs[4], total_protein) / total_protein.sum().item()
		loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss

		return loss, total_calories_loss, total_mass_loss, total_fat_loss, total_carb_loss, total_protein_loss

	def train_step(self, x):
		(loss, total_calories_loss, total_mass_loss, total_fat_loss, total_carb_loss, total_protein_loss), grads = self.grad_fn(x)
		self.optimizer(grads)
		return loss, total_calories_loss, total_mass_loss, total_fat_loss, total_carb_loss, total_protein_loss

	def train_epoch(self, epoch):
		self.net.set_train()
		self.net2.set_train()
		self.net_cat.set_train()

		train_loss = 0
		calories_loss = 0
		mass_loss = 0
		fat_loss = 0
		carb_loss = 0
		protein_loss = 0

		num_batches = self.train_dataset.get_dataset_size()
		data_iterator = tqdm(self.train_dataloader, desc=f"---Epoch {epoch} of training---")
		for batch_idx, x in enumerate(data_iterator):
			loss, total_calories_loss, total_mass_loss, total_fat_loss, total_carb_loss, total_protein_loss = self.train_step(x)

			train_loss += float(loss)
			calories_loss += float(total_calories_loss)
			mass_loss += float(total_mass_loss)
			fat_loss += float(total_fat_loss)
			carb_loss += float(total_carb_loss)
			protein_loss += float(total_protein_loss)

			# if (batch_idx+1) % 100 == 0 or batch_idx+1 == num_batches:
		print('\nEpoch: [{}]\t'
				'Loss: {:2.5f} \t'
				'calorieloss: {:2.5f} \t'
				'massloss: {:2.5f} \t'
				'fatloss: {:2.5f} \t'
				'carbloss: {:2.5f} \t'
				'proteinloss: {:2.5f} \t'
				'lr:{:.7f}\n'.format(
				epoch,
				train_loss/num_batches,
				calories_loss/num_batches,
				mass_loss/num_batches,
				fat_loss/num_batches,
				carb_loss/num_batches,
				protein_loss/num_batches,
				float(self.optimizer.get_lr())))

	def valid(self, epoch):
		self.net.set_train(False)
		self.net2.set_train(False)
		self.net_cat.set_train(False)

		train_loss = 0
		calories_loss = 0
		mass_loss = 0
		fat_loss = 0
		carb_loss = 0
		protein_loss = 0

		num_batches = self.val_dataset.get_dataset_size()
		data_iterator = tqdm(self.val_dataloader, desc=f"---Epoch {epoch} of validation---")
		for batch_idx, x in enumerate(data_iterator):
			loss, total_calories_loss, total_mass_loss, total_fat_loss, total_carb_loss, total_protein_loss = self.forward_fn(x)

			train_loss += float(loss)
			calories_loss += float(total_calories_loss)
			mass_loss += float(total_mass_loss)
			fat_loss += float(total_fat_loss)
			carb_loss += float(total_carb_loss)
			protein_loss += float(total_protein_loss)

			# if (batch_idx+1) % 100 == 0 or batch_idx+1 == num_batches:

		print('\nEpoch: [{}]\t'
			  'Loss: {:2.5f} \t'
			  'calorieloss: {:2.5f} \t'
			  'massloss: {:2.5f} \t'
			  'fatloss: {:2.5f} \t'
			  'carbloss: {:2.5f} \t'
			  'proteinloss: {:2.5f} \t'
			  'lr:{:.7f}\n'.format(
				epoch,
			   train_loss / num_batches,
			   calories_loss / num_batches,
			   mass_loss / num_batches,
			   fat_loss / num_batches,
			   carb_loss / num_batches,
			   protein_loss / num_batches,
			float(self.optimizer.get_lr())))
