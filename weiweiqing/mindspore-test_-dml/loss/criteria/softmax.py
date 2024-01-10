import mindspore
import numpy as np
from mindspore import Tensor, nn, ops, Parameter, dtype
from . import register_criteria
from mindspore.common.initializer import initializer, Uniform

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


@register_criteria
# This Implementation follows: https://github.com/azgo14/classification_metric_learning
class Softmax(nn.Cell):
    def __init__(self, opt, **kwargs):
        super(Softmax, self).__init__()

        self.temperature = opt.loss_softmax_temperature

        self.class_map = Parameter(Tensor(np.random.randn(opt.n_classes, opt.embed_dim), mindspore.float32))
        stdv = 1. / np.sqrt(self.class_map.shape[1])
        # self.class_map.data.uniform_(-stdv, stdv)
        self.class_map.set_data(initializer(Uniform(scale=stdv), self.class_map.shape, self.class_map.dtype))

        self.lr = opt.loss_softmax_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        class_mapped_batch = ops.dense(batch, ops.L2Normalize(axis=1)(self.class_map))

        loss = nn.CrossEntropyLoss()(class_mapped_batch/self.temperature, labels.astype(dtype.int32))  # .to(self.par.device))

        return loss
