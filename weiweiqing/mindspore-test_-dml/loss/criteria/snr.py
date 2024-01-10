from .. import batch_miner
from mindspore import nn, ops
from . import register_criteria


"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batch_miner._dict.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


@register_criteria
# This implements the Signal-To-Noise Ratio Triplet Loss
class Snr(nn.Cell):
    def __init__(self, opt, batch_miner):
        super(Snr, self).__init__()
        self.margin     = opt.loss_snr_margin
        self.reg_lambda = opt.loss_snr_reg_lambda
        self.batch_miner = batch_miner

        if self.batch_miner.name == 'distance':
            self.reg_lambda = 0

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        sampled_triplets = self.batch_miner(batch, labels)
        anchors   = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]

        pos_snr  = ops.var(batch[anchors,:]-batch[positives,:], axis=1)/ops.var(batch[anchors,:], axis=1)
        neg_snr  = ops.var(batch[anchors,:]-batch[negatives,:], axis=1)/ops.var(batch[anchors,:], axis=1)

        reg_loss = ops.mean(ops.abs(ops.sum(batch[anchors,:], dim=1)))

        snr_loss = ops.relu(pos_snr - neg_snr + self.margin)
        snr_loss = ops.sum(snr_loss)/ops.sum(snr_loss > 0)

        loss = snr_loss + self.reg_lambda * reg_loss

        return loss
