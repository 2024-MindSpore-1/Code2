from mindspore import Tensor, nn, ops
from . import register_criteria

"""================================================================================================="""
ALLOWED_MINING_OPS = ['NPair']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


@register_criteria
class NPair(nn.Cell):
    def __init__(self, opt, batch_miner):
        """
        Args:
        """
        super(NPair, self).__init__()
        self.opt = opt

        self.l2_weight = opt.loss_npair_l2
        self.batch_miner = batch_miner

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        anchors, positives, negatives = self.batch_miner(batch, labels)

        ##
        loss  = 0
        if 'bninception' in self.pars.arch:
            ### clamping/value reduction to avoid initial overflow for high embedding dimensions!
            batch = batch/4
        for anchor, positive, negative_set in zip(anchors, positives, negatives):
            a_embs, p_embs, n_embs = batch[anchor:anchor+1], batch[positive:positive+1], batch[negative_set]
            inner_sum = a_embs[:,None,:].bmm((n_embs - p_embs[:,None,:]).permute(0,2,1))
            inner_sum = inner_sum.view(inner_sum.shape[0], inner_sum.shape[-1])
            loss  = loss + ops.mean(ops.log(ops.sum(ops.exp(inner_sum), dim=1) + 1))/len(anchors)
            loss  = loss + self.l2_weight*ops.mean(ops.norm(batch, ord=2, dim=1))/len(anchors)

        return loss
