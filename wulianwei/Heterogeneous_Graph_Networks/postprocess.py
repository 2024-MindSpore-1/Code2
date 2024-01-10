import os
import argparse
import numpy as np

from src.metrics import BGCFEvaluate
from src.dataset import load_graph

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Beauty", help="choose which dataset")
parser.add_argument("--datapath", type=str, default="./scripts/data_mr", help="minddata path")
parser.add_argument('--input_dim', type=int, default=64, choices=[64, 128],
                    help="user and item embedding dimension")
parser.add_argument('--Ks', type=list, default=[5, 10, 20, 100], help="top K")
parser.add_argument('--workers', type=int, default=8, help="number of process to generate data")
parser.add_argument("--result_path", type=str, default="./result_Files", help="result path")
args = parser.parse_args()


def get_acc():
    """calculate accuracy"""
    train_graph, test_graph, _ = load_graph(args.datapath)

    num_user = train_graph.graph_info()["node_num"][0]
    num_item = train_graph.graph_info()["node_num"][1]
    input_dim = args.input_dim
    user_reps = np.zeros([num_user, input_dim * 3])
    item_reps = np.zeros([num_item, input_dim * 3])

    for i in range(50):
        sub_folder = os.path.join(args.result_path, 'result_Files_' + str(i))
        user_rep = np.fromfile(os.path.join(sub_folder, 'amazon-beauty_0.bin'), np.float16)
        user_rep = user_rep.reshape(num_user, input_dim * 3)
        item_rep = np.fromfile(os.path.join(sub_folder, 'amazon-beauty_1.bin'), np.float16)
        item_rep = item_rep.reshape(num_item, input_dim * 3)

        user_reps += user_rep
        item_reps += item_rep
    user_reps /= 50
    item_reps /= 50

    eval_class = BGCFEvaluate(args, train_graph, test_graph, args.Ks)

    test_recall_bgcf, test_ndcg_bgcf, \
    test_sedp, test_nov = eval_class.eval_with_rep(user_reps, item_reps, args)

    print('recall_@10:%.5f,     recall_@20:%.5f,     ndcg_@10:%.5f,    ndcg_@20:%.5f,   '
          'sedp_@10:%.5f,     sedp_@20:%.5f,    nov_@10:%.5f,    nov_@20:%.5f\n' % (test_recall_bgcf[1],
                                                                                    test_recall_bgcf[2],
                                                                                    test_ndcg_bgcf[1],
                                                                                    test_ndcg_bgcf[2],
                                                                                    test_sedp[0],
                                                                                    test_sedp[1],
                                                                                    test_nov[1],
                                                                                    test_nov[2]))


if __name__ == "__main__":
    get_acc()
