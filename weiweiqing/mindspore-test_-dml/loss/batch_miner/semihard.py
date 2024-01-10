import numpy as np
from mindspore import Tensor
from . import register_batch_miner, pdist


@register_batch_miner
class SemiHard:
    def __init__(self, opt):
        self.opt = opt
        self.margin = vars(opt)['loss_'+opt.loss+'_margin']

    def __call__(self, batch, labels, return_distances=False):
        if isinstance(labels, Tensor):
            # labels = labels.detach().numpy()
            labels = labels.asnumpy()

        bs = batch.shape[0]

        # Return distance matrix for all elements in batch (BSxBS)
        distances = pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels != l
            pos = labels == l

            anchors.append(i)
            pos[i] = 0
            p = np.random.choice(np.where(pos)[0])
            positives.append(p)

            # Find negatives that violate triplets constraint semi-negatives
            neg_mask = np.logical_and(neg, d > d[p])
            neg_mask = np.logical_and(neg_mask, d < self.margin+d[p])
            if neg_mask.sum() > 0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets

