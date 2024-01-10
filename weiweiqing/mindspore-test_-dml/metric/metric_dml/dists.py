from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np
from mindspore import Tensor, ops
from . import register_metric


@register_metric
class Dists:
    def __init__(self, mode, **kwargs):
        self.mode        = mode
        self.requires = ['features', 'target_labels']
        self.name     = 'dists@{}'.format(mode)

    def __call__(self, features, target_labels):
        features_locs = []
        for lab in np.unique(target_labels):
            features_locs.append(np.where(target_labels==lab)[0])

        if 'intra' in self.mode:
            if isinstance(features, Tensor):
                intrafeatures = features.asnumpy()
            else:
                intrafeatures = features

            intra_dists = []
            for loc in features_locs:
                c_dists = distance.cdist(intrafeatures[loc], intrafeatures[loc], 'cosine')
                c_dists = np.sum(c_dists)/(len(c_dists)**2-len(c_dists))
                intra_dists.append(c_dists)
            intra_dists = np.array(intra_dists)
            maxval      = np.max(intra_dists[1-np.isnan(intra_dists)])
            intra_dists[np.isnan(intra_dists)] = maxval
            intra_dists[np.isinf(intra_dists)] = maxval
            dist_metric = dist_metric_intra = np.mean(intra_dists)

        if 'inter' in self.mode:
            if not isinstance(features, Tensor):
                coms = []
                for loc in features_locs:
                    com   = normalize(np.mean(features[loc], axis=0).reshape(1, -1)).reshape(-1)
                    coms.append(com)
                mean_inter_dist = distance.cdist(np.array(coms), np.array(coms), 'cosine')
                dist_metric = dist_metric_inter = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))
            else:
                coms = []
                for loc in features_locs:
                    com   = ops.L2Normalize(ops.mean(features[loc], axis=0).reshape(1, -1), axis=-1).reshape(1, -1)
                    coms.append(com)
                mean_inter_dist = 1 - ops.cat(coms, axis=0).mm(ops.cat(coms, axis=0).T).asnumpy()
                dist_metric = dist_metric_inter = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))

        if self.mode=='intra_over_inter':
            dist_metric = dist_metric_intra/np.clip(dist_metric_inter, 1e-8, None)

        return dist_metric
