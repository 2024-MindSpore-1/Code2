from mindspore import Tensor, nn, ops
from . import register_criteria

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False


@register_criteria
class MultiSimilarity(nn.Cell):
    def __init__(self, opt):
        super(MultiSimilarity, self).__init__()
        self.n_classes          = opt.n_classes

        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin     = opt.loss_multisimilarity_margin
        self.thresh     = opt.loss_multisimilarity_thresh

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        similarity = batch.mm(batch.T)

        loss = []
        for i in range(len(batch)):
            pos_idxs       = labels == labels[i]
            pos_idxs[i]    = 0
            neg_idxs       = labels != labels[i]

            anchor_pos_sim = similarity[i][pos_idxs]
            anchor_neg_sim = similarity[i][neg_idxs]

            # This part doesn't really work, especially when you dont have a lot of positives in the batch...
            neg_idxs = (anchor_neg_sim + self.margin) > ops.min(anchor_pos_sim)
            pos_idxs = (anchor_pos_sim - self.margin) < ops.max(anchor_neg_sim)
            if not ops.sum(Tensor(neg_idxs)) or not ops.sum(Tensor(pos_idxs)):
                continue
            anchor_neg_sim = anchor_neg_sim[neg_idxs]
            anchor_pos_sim = anchor_pos_sim[pos_idxs]

            pos_term = 1./self.pos_weight * ops.log(1 + ops.sum(ops.exp(-self.pos_weight* (anchor_pos_sim - self.thresh))))
            neg_term = 1./self.neg_weight * ops.log(1 + ops.sum(ops.exp(self.neg_weight * (anchor_neg_sim - self.thresh))))

            loss.append(pos_term + neg_term)

        loss = ops.mean(ops.stack(loss))

        return loss
