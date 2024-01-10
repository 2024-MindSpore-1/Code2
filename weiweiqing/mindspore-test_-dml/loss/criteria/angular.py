import numpy as np
from mindspore import nn, ops
from . import register_criteria

"""================================================================================================="""
ALLOWED_MINING_OPS = ['NPair']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


@register_criteria
class Angular(nn.Cell):
    def __init__(self, opt, batch_miner):
        super(Angular, self).__init__()

        self.tan_angular_margin = np.tan(np.pi/180*opt.loss_angular_alpha)
        self.lam            = opt.loss_angular_npair_ang_weight
        self.l2_weight      = opt.loss_angular_npair_l2
        self.batch_miner     = batch_miner

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        # NOTE: Normalize Angular Loss, but dont normalize npair loss!
        anchors, positives, negatives = self.batch_miner(batch, labels)
        anchors, positives, negatives = batch[anchors], batch[positives], batch[negatives]
        n_anchors, n_positives, n_negatives = ops.L2Normalize(anchors, axis=1), ops.L2Normalize(positives, axis=1), ops.L2Normalize(negatives, axis=-1)

        is_term1 = 4*self.tan_angular_margin**2*(n_anchors + n_positives)[:, None, :].bmm(n_negatives.permute(0, 2, 1))
        is_term2 = 2*(1+self.tan_angular_margin**2)*n_anchors[:, None, :].bmm(n_positives[:, None, :].permute(0, 2, 1))
        is_term1 = is_term1.view(is_term1.shape[0], is_term1.shape[-1])
        is_term2 = is_term2.view(-1, 1)

        inner_sum_ang = is_term1 - is_term2
        angular_loss = ops.mean(ops.log(ops.sum(ops.exp(inner_sum_ang), dim=1) + 1))

        inner_sum_npair = anchors[:, None, :].bmm((negatives - positives[:, None, :]).permute(0, 2, 1))
        inner_sum_npair = inner_sum_npair.view(inner_sum_npair.shape[0], inner_sum_npair.shape[-1])
        npair_loss = ops.mean(ops.log(ops.sum(ops.exp(inner_sum_npair.clamp(max=50, min=-50)), dim=1) + 1))

        loss = npair_loss + self.lam*angular_loss + self.l2_weight*ops.mean(ops.norm(batch, ord=2, dim=1))
        return loss
