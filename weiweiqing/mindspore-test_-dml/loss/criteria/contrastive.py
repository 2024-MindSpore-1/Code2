from .. import batch_miner
from mindspore import nn, ops, Tensor
from . import register_criteria

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batch_miner._dict.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


@register_criteria
class Contrastive(nn.Cell):
    def __init__(self, opt, batch_miner, **kwargs):
        super(Contrastive, self).__init__()
        self.pos_margin = opt.loss_contrastive_pos_margin
        self.neg_margin = opt.loss_contrastive_neg_margin
        self.batch_miner = batch_miner
        self.stable = opt.stable

        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        sampled_triplets = self.batch_miner(batch, labels)

        anchors   = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]

        # pos_dists = ops.mean(ops.relu(nn.PairwiseDistance(p=2)(batch[anchors, :], batch[positives, :]) - self.pos_margin))
        # neg_dists = ops.mean(ops.relu(self.neg_margin - nn.PairwiseDistance(p=2)(batch[anchors, :], batch[negatives, :])))

        pos_dists = ops.mean(ops.relu(Tensor([float(ops.norm(batch[anchor] - batch[positive])) - self.pos_margin for positive in positives for anchor in anchors])))
        if self.stable:
            pos_dists = ops.log(1 + pos_dists)
            # pos_dists = torch.sqrt(pos_dists)
        neg_dists = ops.mean(ops.relu(Tensor([float(ops.norm(batch[anchor] - batch[negative])) - self.neg_margin for negative in negatives for anchor in anchors])))

        loss      = pos_dists + neg_dists

        return loss
