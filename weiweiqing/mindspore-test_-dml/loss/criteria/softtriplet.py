import numpy as np
from mindspore import nn, Tensor, ops, dtype, Parameter
from mindspore.common.initializer import initializer, Uniform
from . import register_criteria

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


@register_criteria
# This implementation follows https://github.com/idstcv/SoftTriple
class SoftTriplet(nn.Cell):
    def __init__(self, opt):
        super(SoftTriplet, self).__init__()

        ####
        self.n_classes   = opt.n_classes

        ####
        self.n_centroids  = opt.loss_softtriplet_n_centroids
        self.margin_delta = opt.loss_softtriplet_margin_delta
        self.gamma        = opt.loss_softtriplet_gamma
        self.lam          = opt.loss_softtriplet_lambda
        self.reg_weight   = opt.loss_softtriplet_reg_weight


        ####
        self.reg_norm    = self.n_classes*self.n_centroids*(self.n_centroids-1)
        self.reg_indices = ops.zeros((self.n_classes*self.n_centroids, self.n_classes*self.n_centroids), dtype=dtype.bool_)  # .to(opt.device)
        for i in range(0, self.n_classes):
            for j in range(0, self.n_centroids):
                self.reg_indices[i*self.n_centroids+j, i*self.n_centroids+j+1:(i+1)*self.n_centroids] = 1


        ####
        self.intra_class_centroids = Parameter(Tensor(opt.embed_dim, self.n_classes*self.n_centroids).to(dtype=dtype.float32))
        stdv = 1. / np.sqrt(self.intra_class_centroids.shape[1])
        # self.intra_class_centroids.data.uniform_(-stdv, stdv)
        self.intra_class_centroids.set_data(initializer(Uniform(scale=stdv), self.intra_class_centroids.shape, self.intra_class_centroids.dtype))
        self.lr   = opt.lr*opt.loss_softtriplet_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        bs = batch.shape[0]

        intra_class_centroids     = ops.L2Normalize(self.intra_class_centroids, axis=1)
        similarities_to_centroids = batch.mm(intra_class_centroids).reshape(-1, self.n_classes, self.n_centroids)

        soft_weight_over_centroids = nn.Softmax(axis=1)(self.gamma*similarities_to_centroids)
        per_class_embed            = ops.sum(soft_weight_over_centroids * similarities_to_centroids, dim=2)

        margin_delta = ops.zeros(per_class_embed.shape)  # .to(self.par.device)
        margin_delta[ops.arange(0, bs), labels] = self.margin_delta

        centroid_classification_loss = nn.CrossEntropyLoss()(self.lam*(per_class_embed-margin_delta), labels.to(dtype.int64))  # .to(self.par.device))

        inter_centroid_similarity = self.intra_class_centroids.T.mm(self.intra_class_centroids)
        regularisation_loss = ops.sum(ops.sqrt(2.00001-2*inter_centroid_similarity[self.reg_indices]))/self.reg_norm

        return centroid_classification_loss + self.reg_weight * regularisation_loss
