import numpy as np
import faiss
# import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
import copy
from . import get_metric
from mindspore import ops, Tensor


class MetricComputer:
    def __init__(self, metric_names, opt):
        self.pars            = opt
        self.metric_names    = metric_names
        self.list_of_metrics = [get_metric(metric_name, opt) for metric_name in metric_names]
        self.requires        = [metric.requires for metric in self.list_of_metrics]
        self.requires        = list(set([x for y in self.requires for x in y]))

    def compute_standard(self, opt, model, dataloader, dataset, evaltypes):   # , device, **kwargs):
        evaltypes = copy.deepcopy(evaltypes)

        n_classes = opt.n_classes
        model.set_train(False)

        feature_colls  = { key: [] for key in evaltypes }

        ###
        target_labels = []
        final_iter = tqdm(dataloader, desc='Embedding Data...'.format(len(evaltypes)))
        for idx, inp in enumerate(final_iter):
            input_img, target = inp[1], inp[0]
            target_labels.extend(target.asnumpy().tolist())
            out = model(input_img)
            if isinstance(out, tuple):
                if self.pars.multi_loss:
                    out_cls, out_patch = out
                    out = self.pars.fuse * out_cls + (1-self.pars.fuse)*out_patch
                else:
                    out, aux_f = out

            # Include embeddings of all output features
            for evaltype in evaltypes:
                if isinstance(out, dict):
                    feature_colls[evaltype].extend(out[evaltype].asnumpy().tolist())
                else:
                    feature_colls[evaltype].extend(out.asnumpy().tolist())

        target_labels = np.hstack(target_labels).reshape(-1, 1)

        computed_metrics = {evaltype: {} for evaltype in evaltypes}
        extra_infos      = {evaltype: {} for evaltype in evaltypes}


        ###
        faiss.omp_set_num_threads(self.pars.kernels)
        # faiss.omp_set_num_threads(self.pars.kernels)
        res = None
        # torch.cuda.empty_cache()
        if self.pars.evaluate_on_gpu:
            res = faiss.StandardGpuResources()

        import time
        for evaltype in evaltypes:
            features        = np.vstack(feature_colls[evaltype]).astype('float32')
            features_cosine = normalize(features, axis=1)


            """============ Compute k-Means ==============="""
            if 'kmeans' in self.requires:
                ### Set CPU Cluster index
                cluster_idx = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans            = faiss.Clustering(features.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                ### Train Kmeans
                kmeans.train(features, cluster_idx)
                centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])

            if 'kmeans_cosine' in self.requires:
                ### Set CPU Cluster index
                cluster_idx = faiss.IndexFlatL2(features_cosine.shape[-1])
                if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans            = faiss.Clustering(features_cosine.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                ### Train Kmeans
                kmeans.train(features_cosine, cluster_idx)
                centroids_cosine = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features_cosine.shape[-1])
                centroids_cosine = normalize(centroids, axis=1)


            """============ Compute Cluster Labels ==============="""
            if 'kmeans_nearest' in self.requires:
                faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids)
                _, computed_cluster_labels = faiss_search_index.search(features, 1)

            if 'kmeans_nearest_cosine' in self.requires:
                faiss_search_index = faiss.IndexFlatIP(centroids_cosine.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids_cosine)
                _, computed_cluster_labels_cosine = faiss_search_index.search(features_cosine, 1)



            """============ Compute Nearest Neighbours ==============="""
            if 'nearest_features' in self.requires:
                faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(features)

                max_kval            = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'Recall' in x])
                _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))
                k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

            if 'nearest_features_cosine' in self.requires:
                faiss_search_index  = faiss.IndexFlatIP(features_cosine.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(normalize(features_cosine,axis=1))

                max_kval                   = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'Recall' in x])
                _, k_closest_points_cosine = faiss_search_index.search(normalize(features_cosine,axis=1), int(max_kval+1))
                k_closest_classes_cosine   = target_labels.reshape(-1)[k_closest_points_cosine[:,1:]]



            ###
            if self.pars.evaluate_on_gpu:
                features        = Tensor.from_numpy(features)  # .to(self.pars.device)
                features_cosine = Tensor.from_numpy(features_cosine)  # .to(self.pars.device)

            start = time.time()
            for metric in self.list_of_metrics:
                input_dict = {}
                if 'features' in metric.requires:         input_dict['features'] = features
                if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels

                if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
                if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
                if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes

                if 'features_cosine' in metric.requires:         input_dict['features_cosine'] = features_cosine

                if 'kmeans_cosine' in metric.requires:           input_dict['centroids_cosine'] = centroids_cosine
                if 'kmeans_nearest_cosine' in metric.requires:   input_dict['computed_cluster_labels_cosine'] = computed_cluster_labels_cosine
                if 'nearest_features_cosine' in metric.requires: input_dict['k_closest_classes_cosine'] = k_closest_classes_cosine

                computed_metrics[evaltype][metric.name] = metric(**input_dict)

            extra_infos[evaltype] = {  'features':features, 'target_labels':target_labels,
                                     'image_paths': dataset.image_paths,
                                     'query_image_paths':None, 'gallery_image_paths':None}

        # torch.cuda.empty_cache()
        return computed_metrics, extra_infos
