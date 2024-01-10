import numpy as np
from mindspore import Tensor
from . import register_batch_miner


@register_batch_miner
class IntraRandom:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, batch, labels):
        if isinstance(labels, Tensor):
            # labels = labels.detach().cpu().numpy()
            labels = labels.asnumpy()
        unique_classes = np.unique(labels)
        indices = np.arange(len(batch))
        class_dict = {i: indices[labels == i] for i in unique_classes}

        sampled_triplets = []
        for cls in np.random.choice(list(class_dict.keys()), len(labels), replace=True):
            a, p, n = np.random.choice(class_dict[cls], 3, replace=True)
            sampled_triplets.append((a, p, n))

        return sampled_triplets
