from .. import batch_miner
from mindspore import Tensor, nn, ops
from . import register_criteria

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batch_miner._dict.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


@register_criteria
# Standard Triplet Loss, finds triplets in Mini-batches.
class Triplet(nn.Cell):
    def __init__(self, opt, batch_miner):
        super(Triplet, self).__init__()
        self.margin     = opt.loss_triplet_margin
        self.batch_miner = batch_miner

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def triplet_distance(self, anchor, positive, negative):
        return ops.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels, **kwargs):
        if isinstance(labels, Tensor):
            labels = labels.asnumpy()
        sampled_triplets = self.batch_miner(batch, labels)
        loss = ops.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return ops.mean(loss)
