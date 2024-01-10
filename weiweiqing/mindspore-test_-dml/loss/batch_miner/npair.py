import numpy as np
from mindspore import Tensor
from . import register_batch_miner


@register_batch_miner
class NPair:
	def __init__(self, opt):
		self.par = opt

	def __call__(self, batch, labels):
		if isinstance(labels, Tensor):
			# labels = labels.detach().cpu().numpy()
			labels = labels.asnumpy()

		anchors, positives, negatives = [], [], []

		for i in range(len(batch)):
			anchor = i
			pos = labels == labels[anchor]

			if np.sum(pos) > 1:
				anchors.append(anchor)
				avail_positive = np.where(pos)[0]  # all positive indices
				avail_positive = avail_positive[avail_positive != anchor]  # all positive except current anchor
				positive = np.random.choice(avail_positive)  # randomly choose a positive sample
				positives.append(positive)  # for each anchor, just choose one positive sample

		negatives = []
		for anchor, positive in zip(anchors, positives):
			neg_idxs = [i for i in range(len(batch)) if i not in [anchor, positive] and labels[i] != labels[anchor]]  # all negative indices
			# neg_idxs = [i for i in range(len(batch)) if i not in [anchor, positive]]
			negative_set = np.arange(len(batch))[neg_idxs]
			negatives.append(negative_set)  # for each anchor, choose all negative samples, negatives: [[],...]

		return anchors, positives, negatives
