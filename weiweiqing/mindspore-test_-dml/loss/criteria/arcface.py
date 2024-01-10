import numpy as np
from mindspore import nn, ops, Parameter, Tensor
from . import register_criteria

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


@register_criteria
# This implementation follows the pseudocode provided in the original paper.
class ArcFace(nn.Cell):
    def __init__(self, opt):
        super(ArcFace, self).__init__()
        self.opt = opt

        ####
        self.angular_margin = opt.loss_arcface_angular_margin
        self.feature_scale  = opt.loss_arcface_feature_scale

        self.class_map = Parameter(Tensor(opt.n_classes, opt.embed_dim))
        stdv = 1. / np.sqrt(self.class_map.size(1))
        self.class_map.data.uniform_(-stdv, stdv)

        self.lr    = opt.loss_arcface_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        # bs, labels = len(batch), labels.to(self.opt.device)
        bs = len(batch)

        class_map      = ops.L2Normalize(self.class_map, axis=1)
        # Note that the similarity becomes the cosine for normalized embeddings. Denoted as 'fc7' in the paper pseudocode.
        cos_similarity = batch.mm(class_map.T).clamp(min=1e-10, max=1-1e-10)

        # pick = ops.zeros(bs, self.opt.n_classes).bool().to(self.opt.device)
        pick = ops.zeros(bs, self.opt.n_classes).bool()
        pick[ops.arange(bs), labels] = 1

        original_target_logit  = cos_similarity[pick]

        theta                 = ops.acos(original_target_logit)
        marginal_target_logit = ops.cos(theta + self.angular_margin)

        class_pred = self.feature_scale * (cos_similarity + pick * (marginal_target_logit-original_target_logit).unsqueeze(1))
        loss       = nn.CrossEntropyLoss()(class_pred, labels)

        return loss
