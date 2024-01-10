import numpy as np
from mindspore import Tensor, ops
from . import pdist, register_batch_miner

# -----------------------------------------------------------------------------------------------------------
# The following code is borrowed from https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
# ----------------------------------------------------------------------------------------------------------


# something wrong???
@register_batch_miner
class Distance:
	def __init__(self, opt):
		self.opt = opt
		self.lower_cutoff = opt.miner_distance_lower_cutoff
		self.upper_cutoff = opt.miner_distance_upper_cutoff

	def __call__(self, batch, labels, tar_labels=None, return_distances=False, distances=None):
		if isinstance(labels, Tensor):
			# labels = labels.detach().cpu().numpy()
			labels = labels.asnumpy()
		# bs, dim = batch.shapeill
		bs, dim = batch.shape[0], batch.shape[-1]

		if distances is None:
			# distances = pdist(batch.detach()).clamp(min=self.lower_cutoff)
			distances = pdist(batch).clamp(min=self.lower_cutoff)
		sel_d = distances.shape[-1]

		positives, negatives = [], []
		labels_visited = []
		anchors = []

		# tar_labels = labels if tar_labels is None else tar_labels
		tar_labels = tar_labels or labels

		for i in range(bs):
			neg = tar_labels != labels[i]
			pos = tar_labels == labels[i]

			anchors.append(i)
			q_d_inv = self.inverse_sphere_distances(dim, bs, distances[i], tar_labels, labels[i])
			negatives.append(np.random.choice(sel_d, p=q_d_inv))

			if np.sum(pos) > 0:
				# Sample positives randomly
				if np.sum(pos) > 1:
					pos[i] = 0
				positives.append(np.random.choice(np.where(pos)[0]))
				# Sample negatives by distance

		sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

		if return_distances:
			return sampled_triplets, distances
		else:
			return sampled_triplets

	@staticmethod
	def inverse_sphere_distances(dim, bs, anchor_to_all_dists, labels, anchor_label):
			dists = anchor_to_all_dists

			# negated log-distribution of distances of unit sphere in dimension <dim>
			log_q_d_inv = ((2.0 - float(dim)) * ops.log(dists) - (float(dim-3) / 2) * ops.log(1.0 - 0.25 * (dists.pow(2))))
			log_q_d_inv[np.where(labels == anchor_label)[0]] = 0

			q_d_inv = ops.exp(log_q_d_inv - ops.max(log_q_d_inv))  # - max(log) for stability
			q_d_inv[np.where(labels == anchor_label)[0]] = 0

			# NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
			# errors where there are no available negatives (for high samples_per_class cases).
			# q_d_inv[np.where(dists.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

			q_d_inv = q_d_inv/q_d_inv.sum()
			return q_d_inv.detach().cpu().numpy()
