import numpy as np
from scipy.special import comb, binom
from mindspore import Tensor
from . import register_metric


@register_metric
class F1:
    def __init__(self, **kwargs):
        self.requires = ['kmeans', 'kmeans_nearest', 'features', 'target_labels']
        self.name     = 'f1'

    def __call__(self, target_labels, computed_cluster_labels, features, centroids):

        N = len(target_labels)

        # region computer cluster labels

        if isinstance(features, Tensor):
            features = features.asnumpy()
        d = np.zeros(len(features))
        for i in range(len(features)):
            d[i] = np.linalg.norm(features[i,:] - centroids[computed_cluster_labels[i],:])

        # map each sample to unique id ( one index with minimum d[index] )
        # for example, sample 1, 4, 8 with computer_cluster[1, 4, 8] = 3, and d[1] = 5, d[4] = 3, d[8] = 6
        # then labels_pred[1/4/8] = 4
        # that is, 4 is Class Identity of sample 1, 4, 8
        # just arrange a temporary Class Identity(Sample No with minimum distance to the corresponding centroid) for all samples of a class
        labels_pred = np.zeros(len(features))
        for i in np.unique(computed_cluster_labels):  # i = 3
            index = np.where(computed_cluster_labels == i)[0]  # index = [1, 4, 8]
            ind = np.argmin(d[index])  # d[index] = [5, 3, 6]  ind = 1
            cid = index[ind]  # cid = index[1] = 4
            labels_pred[index] = cid  # labels_pred[1,4,8] = 4

        # for example: labels_pred[1, 4, 8] = 4, labels_pred[0, 2, 7] = 7, labels_pred[3, 5, 6] = 5
        keys     = np.unique(labels_pred)   # keys = [7, 4, 5]
        num_item = len(keys)    # num_item = 3
        values   = range(num_item)  # values = [0, 1, 2]
        item_map = dict()
        for i in range(len(keys)):
            item_map.update([(keys[i], values[i])])  # item_map = { 7: 0, 4: 1, 5: 2 }
        # that is: two kinds of Class Identity for each sample, for example, 4 or 1 is Class Identity of sample 1, 4, 8
        # the second Class Identity is better considering that the sample with smaller no will has smaller Class No as possible

        count_item = np.zeros(num_item)  # count_item = [0, 0, 0]
        for i in range(N):  # N = 9
            index = item_map[labels_pred[i]]  # get class no for sample i
            count_item[index] = count_item[index] + 1
        # at last, count_itme[0] = 3, count_item[1] = 3, count_item[2] = 3
        # that is, count the number of samples for each class with the second kind of class identity

        # endregion

        # region target labels

        # for example, sample[0, 1, 2, 3, 4, 5, 6, 7, 8]
        # with target labels [0, 1, 0, 2, 1, 2, 2, 0, 1]
        # cluster n_labels
        avail_labels = np.unique(target_labels)  # [0, 1, 2]
        n_labels     = len(avail_labels)  # 3

        # count the number of objects in each cluster
        count_cluster = np.zeros(n_labels)  # [0, 0, 0]
        for i in range(n_labels):
            count_cluster[i] = len(np.where(target_labels == avail_labels[i])[0])
        # at last, count_cluster[0] = 3, count_cluster[1] = 3, count_cluster[2] = 3

        # endregion

        # region compute True Positive (TP) , False Positive (FP), False Negative (FN)

        # method: check each pair, if two samples of the pair have the same computer label, tp =+ 1, otherwise fp =+ 1
        tp_fp = comb(count_cluster, 2).sum()
        # for above example, 3 + 3 + 3 = 9
        # class 0: 1 4 8; class 1: 0 2 7; class 2: 3 5 6
        # all pairs:
        # (1, 4) (1, 8) (4, 8)
        # (0, 2) (0, 7) (2, 7)
        # (3, 5) (3, 6) (5, 6)

        # compute True Positive (TP)
        tp     = 0
        for k in range(n_labels):  # n_labels = 3
            member     = np.where(target_labels == avail_labels[k])[0]  # for example, when class no k = 1, member = [0, 2, 7]
            member_ids = labels_pred[member]  # member_ids = [7, 7, 7], note: in this example, predictions are 100% correct, generally not
            count = np.zeros(num_item)  # count = [0, 0, 0]
            for j in range(len(member)):  # j = 0, 1, 2 for each member sample
                index = item_map[member_ids[j]]   # index = 0, 0, 0
                count[index] = count[index] + 1
            # at last, count[0] = 3, count[1] = 0, count[2] = 0, note: in this example, predictions are 100% correct, generally not

            # for each pair, if two samples of the pair in the same count[computer_class_no], tp += 1
            tp += comb(count, 2).sum()  # tp += 3, note: in this example, predictions are 100% correct, generally not

        # False Positive (FP)
        fp = tp_fp - tp

        # Compute False Negative (FN)
        count = comb(count_item, 2).sum()
        fn = count - tp

        # endregion

        # region compute F measure

        P = tp / (tp + fp)
        R = tp / (tp + fn)
        beta = 1
        F = (beta*beta + 1) * P * R / (beta*beta * P + R)
        return F

        # endregion
