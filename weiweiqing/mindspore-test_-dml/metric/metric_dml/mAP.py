from mindspore import Tensor
import numpy as np
import faiss
from . import register_metric


@register_metric
class MAP():
    def __init__(self, **kwargs):
        self.requires = ['features', 'target_labels']
        self.name     = 'mAP'

    def __call__(self, target_labels, features):
        # for example, samples [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #        target_labels:[0, 1, 2, 0, 1, 2, 0, 1, 2]
        labels, freqs = np.unique(target_labels, return_counts=True)  # labels = [0, 1, 2], frequencies = [3, 3, 3]
        R             = len(features)   # R = 9, the number of samples

        faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
        if isinstance(features, Tensor):
            features = features.asnumpy()
            res = faiss.StandardGpuResources()
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)        
        faiss_search_index.add(features)
        nearest_neighbours  = faiss_search_index.search(features, int(R+1))[1][:, 1:]   # [R, R], nn indices matrix
        # [R, R] nearest neighbors indices, the 0th column is the nearest neighbor and the last column is -1

        target_labels = target_labels.reshape(-1)
        nn_labels = target_labels[nearest_neighbours]  # nn_labels: [R, R], nn target labels matrix

        avg_r_precisions = []
        # cycle for each class
        for label, freq in zip(labels, freqs):   # each label, frequency of each class, for example, label = 1, freqs = 3
            rows_with_label = np.where(target_labels==label)[0]   # [1, 4, 7]  sample indices with label no
            # cycle for each sample belong to the current class
            for row in rows_with_label:  # row = 1, 4, 7, for example, row = 1
                n_recalled_samples           = np.arange(1, R+1)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

                target_label_occ_in_row      = nn_labels[row, :] == label  # target label occurred in row
                # like [true, true, false, true, false, true, false, false, false]
                # equals to [1, 1, 0, 1, 0, 1, 0, 0, 0]
                # that is, sample 1 has the same label with the first 3 nearest neighbors

                cumsum_target_label_freq_row = np.cumsum(target_label_occ_in_row)  # cumulative sum [1, 2, 2, 3, 3, 4, 4, 4, 4]

                avg_r_pr_row = np.sum(cumsum_target_label_freq_row*target_label_occ_in_row/n_recalled_samples)/freq  # average r per row
                # cumsum_target_label_freq_row*target_label_occ_in_row
                #   [1, 2, 2, 3, 3, 4, 4, 4, 4]
                # * [1, 1, 0, 1, 0, 1, 0, 0, 0]
                # = [1, 2, 0,  3,   0,  4  , 0, 0, 0] / n_recalled_samples, that is
                # / [1, 2, 3,  4,   5,  6  , 7, 8, 9]
                # = [1, 1, 0, 0.75, 0, 0.67, 0, 0, 0] => sum / freq(3)
                # = (1 + 1 + 0.75 + 0.67) / 3

                avg_r_precisions.append(avg_r_pr_row)

        return np.mean(avg_r_precisions)
