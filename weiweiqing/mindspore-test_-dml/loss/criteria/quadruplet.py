import numpy as np
from mindspore import nn, ops
from .. import batch_miner
from . import register_criteria

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batch_miner._dict.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


@register_criteria
class QuadRuplet(nn.Cell):
    def __init__(self, opt, batch_miner):
        super(QuadRuplet, self).__init__()
        self.batch_miner = batch_miner

        self.margin_alpha_1 = opt.loss_quadruplet_margin_alpha_1
        self.margin_alpha_2 = opt.loss_quadruplet_margin_alpha_2

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def triplet_distance(self, anchor, positive, negative):
        return ops.relu(ops.norm(anchor-positive, ord=2, dim=-1)-ops.norm(anchor-negative, ord=2, dim=-1)+self.margin_alpha_1)

    def quadruplet_distance(self, anchor, positive, negative, fourth_negative):
        return ops.relu(ops.norm(anchor-positive, ord=2, dim=-1)-ops.norm(negative-fourth_negative, ord=2, dim=-1)+self.margin_alpha_2)

    def construct(self, batch, labels, **kwargs):
        sampled_triplets    = self.batch_miner(batch, labels)

        anchors   = np.array([triplet[0] for triplet in sampled_triplets]).reshape(-1,1)
        positives = np.array([triplet[1] for triplet in sampled_triplets]).reshape(-1,1)
        negatives = np.array([triplet[2] for triplet in sampled_triplets]).reshape(-1,1)

        fourth_negatives = negatives != negatives.T
        fourth_negatives = [np.random.choice(np.arange(len(batch))[idxs]) for idxs in fourth_negatives]

        triplet_loss     = self.triplet_distance(batch[anchors,:],batch[positives,:],batch[negatives,:])
        quadruplet_loss  = self.quadruplet_distance(batch[anchors,:],batch[positives,:],batch[negatives,:],batch[fourth_negatives,:])

        return ops.mean(triplet_loss) + ops.mean(quadruplet_loss)
