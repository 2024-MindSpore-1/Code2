# The code in the package has been adapted from https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch


_dict = {}


def register_criteria(cls):
    if cls.__name__ in _dict:
        raise ValueError(f"Cannnot register duplicated criteria: {cls.__name__}")
    _dict[cls.__name__] = cls
    return cls


def get_criterion(criteria_name, opt, to_optim, batch_miner=None):
    if criteria_name not in _dict:
        raise ValueError(f"Cannot find Criteria: {criteria_name}")

    # if criteria.REQUIRES_BATCHMINER:
    #     if batch_miner is None:
    #         raise Exception('Loss {} requires one of the following batch mining methods: {}'.format(criteria_name, criteria.ALLOWED_MINING_OPS))
    #     else:
    #         if batch_miner.__class__.__name__ not in criteria.ALLOWED_MINING_OPS:
    #             raise Exception('{}-mining not allowed for {}-loss!'.format(batch_miner.__class__.__name__, criteria_name))

    criteria_par_dict = {'opt': opt, 'batch_miner': batch_miner}
    # # if criteria.REQUIRES_BATCHMINER:
    # criteria_par_dict['batch_miner'] = batch_miner

    criteria = _dict[criteria_name](**criteria_par_dict)

    if criteria.REQUIRES_OPTIM:
        if hasattr(criteria, 'optim_dict_list') and criteria.optim_dict_list is not None:
            to_optim += criteria.optim_dict_list
        else:
            to_optim += [{'params': criteria.trainable_params(), 'lr': criteria.lr}]

    return criteria, to_optim


from .adversarial_separation import AdversarialSeparation
from .angular import Angular
from .arcface import ArcFace
from .contrastive import Contrastive
from .histogram import Histogram
from .margin import Margin
from .multisimilarity import MultiSimilarity
from .npair import NPair
from .proxynca import ProxyNca
from .quadruplet import QuadRuplet
from .snr import Snr
from .softmax import Softmax
from .softtriplet import SoftTriplet
from .triplet import Triplet
