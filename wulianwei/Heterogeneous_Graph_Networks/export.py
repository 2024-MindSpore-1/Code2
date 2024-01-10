import os
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint

from src.bgcf import BGCF
from src.callback import ForwardBGCF

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''run export.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)

    num_user, num_item = 7068, 3570

    network = BGCF([config.input_dim, num_user, num_item],
                   config.embedded_dimension,
                   config.activation,
                   [0.0, 0.0, 0.0],
                   num_user,
                   num_item,
                   config.input_dim)

    load_checkpoint(config.ckpt_file, net=network)

    forward_net = ForwardBGCF(network)

    users = Tensor(np.zeros([num_user,]).astype(np.int32))
    items = Tensor(np.zeros([num_item,]).astype(np.int32))
    neg_items = Tensor(np.zeros([num_item, 1]).astype(np.int32))
    u_test_neighs = Tensor(np.zeros([num_user, config.row_neighs]).astype(np.int32))
    u_test_gnew_neighs = Tensor(np.zeros([num_user, config.gnew_neighs]).astype(np.int32))
    i_test_neighs = Tensor(np.zeros([num_item, config.row_neighs]).astype(np.int32))
    i_test_gnew_neighs = Tensor(np.zeros([num_item, config.gnew_neighs]).astype(np.int32))

    input_data = [users, items, neg_items, u_test_neighs, u_test_gnew_neighs, i_test_neighs, i_test_gnew_neighs]
    export(forward_net, *input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == "__main__":
    run_export()
