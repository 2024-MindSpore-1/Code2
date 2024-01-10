import numpy as np
from mindspore import Tensor, ops
from . import register_batch_miner, pdist


@register_batch_miner
class RandomDistance:
    def __init__(self, opt):
        self.opt = opt
        self.lower_cutoff = opt.miner_distance_lower_cutoff
        self.upper_cutoff = opt.miner_distance_upper_cutoff

    def __call__(self, batch, labels):
        if isinstance(labels, Tensor):
            # labels = labels.detach().cpu().numpy()
            labels = labels.asnumpy()
        labels = labels[np.random.choice(len(labels), len(labels), replace=False)]

        bs = batch.shape[0]
        distances = pdist(batch).clamp(min=self.lower_cutoff)

        positives, negatives = [], []
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels != labels[i]
            pos = labels == labels[i]

            if np.sum(pos) > 1:
                anchors.append(i)
                q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
                # Sample positives randomly
                pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))
                # Sample negatives by distance
                negatives.append(np.random.choice(bs, p=q_d_inv))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    @staticmethod
    def inverse_sphere_distances(batch, anchor_to_all_dists, labels, anchor_label):
            dists = anchor_to_all_dists
            bs, dim = len(dists), batch.shape[-1]

            # negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * ops.log(dists) - (float(dim-3) / 2) * ops.log(1.0 - 0.25 * (dists.pow(2))))
            log_q_d_inv[np.where(labels == anchor_label)[0]] = 0

            q_d_inv = ops.exp(log_q_d_inv - ops.max(log_q_d_inv))  # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            # NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
            # errors where there are no available negatives (for high samples_per_class cases).
            # q_d_inv[np.where(dists.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

