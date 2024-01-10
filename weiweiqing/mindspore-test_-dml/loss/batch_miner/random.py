import numpy as np
import itertools as it
import random
from mindspore import Tensor
from . import register_batch_miner


@register_batch_miner
class Random:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, batch, labels):
        if isinstance(labels, Tensor):
            # labels = labels.detach().cpu().numpy()
            labels = labels.asnumpy()
        unique_classes = np.unique(labels)
        indices = np.arange(len(batch))
        class_dict = {i: indices[labels == i] for i in unique_classes}

        sampled_triplets = [list(it.product([x], [x], [y for y in unique_classes if x != y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        sampled_triplets = [
            [x for x in list(it.product(*[class_dict[j] for j in i])) if x[0] != x[1]]
            for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        # NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
        return sampled_triplets
