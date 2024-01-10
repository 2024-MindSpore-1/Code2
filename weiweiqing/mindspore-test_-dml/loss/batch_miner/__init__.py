# The code in the package has been adapted from https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch

from mindspore import ops
import numpy as np

_dict = {}


def register_batch_miner(cls):
	if cls.__name__ in _dict:
		raise ValueError("Cannot register duplicated batch miner")
	_dict[cls.__name__] = cls
	return cls


def get_batch_miner(batch_miner_name, opt):
	if batch_miner_name not in _dict:
		raise ValueError(f"Cannot find batch miner: {batch_miner_name}")
	return _dict[batch_miner_name](opt)


def pdist(A):
	prod = ops.matmul(A, A.t())
	norm = prod.diagonal().unsqueeze(1).expand_as(prod)
	res = (norm + norm.t() - 2 * prod).clamp(min=0)
	return res.clamp(min=1e-4).sqrt()


from .distance import Distance
from .intra_random import IntraRandom
from .lifted import Lifted
from .npair import NPair
from .parametric import Parametric
from .random import Random
from .random_distance import RandomDistance
from .rho_distance import RhoDistance
from .semihard import SemiHard
from .softhard import SoftHard
