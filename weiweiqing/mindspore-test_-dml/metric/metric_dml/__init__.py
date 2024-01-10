# The code in the package has been adapted from https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch


_dict = {}


def register_metric(cls):
    if cls.__name__ in _dict:
        raise ValueError(f"Cannot register duplicated metric: {cls.__name__}")
    _dict[cls.__name__] = cls
    return cls


def get_metric(metric_name, opt):
    if "E_Recall" in metric_name or "C_Recall" in metric_name:
        name = metric_name.split('@')[0]
        k = int(metric_name.split('@')[-1])
        return _dict[name](k)
    elif "Dists" in metric_name:
        name = metric_name.split('@')[0]
        mode = metric_name.split('@')[-1]
        return _dict[name](mode)
    elif "Rho_Spectrum" in metric_name:
        name = metric_name.split('@')[0]
        mode = int(metric_name.split('@')[-1])
        embed_dim = opt.rho_spectrum_embed_dim
        return _dict[name](embed_dim, mode=mode, opt=opt)
    elif metric_name in _dict:
        return _dict[metric_name]()
    else:
        raise NotImplementedError(f"Metric {metric_name} not available!")


# Metrics based on euclidean distances
from .e_recall import E_Recall
from .nmi import NMI
from .mAP import MAP
from .mAP_c import  MAP_C
from .mAP_lim import MAP_Lim
from .mAP_1000 import MAP_1000
from .f1 import F1

# Metrics based on cosine similarity
from .c_recall import C_Recall
from .c_nmi import C_NMI
from .c_mAP_c import C_mAP_C
from .c_mAP_1000 import C_mAP_1000
from .c_mAP_lim import C_mAP_Lim
from .c_f1 import C_F1

# Generic Embedding space metrics
from .dists import Dists
from .rho_spectrum import Rho_Spectrum




