# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torch
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from src.cuda import CUDA
import mindspore
import mindspore as ms
import mindspore.nn as nn


# class LSTMEncoder(nn.Module):
# 修改
from mindspore import tensor, Tensor


class LSTMEncoder(nn.Cell):
    """ simple wrapper for a bi-lstm """
    def __init__(self, emb_dim, hidden_dim, layers, bidirectional, dropout, pack=True):
        super(LSTMEncoder, self).__init__()

        self.num_directions = 2 if bidirectional else 1

        # self.lstm = nn.LSTM(
        #修改
        self.lstm = mindspore.nn.LSTM(
            emb_dim,
            hidden_dim // self.num_directions,
            layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout)

        self.pack = pack

    def init_state(self, input):
        # batch_size = input.size(0) # retrieve dynamically for decoding
        # 修改
        batch_size = input.shape[0]  # retrieve dynamically for decoding
        # h0 = Variable(torch.zeros(
        #     self.lstm.num_layers * self.num_directions,
        #     batch_size,
        #     self.lstm.hidden_size
        # ), requires_grad=False)
        # c0 = Variable(torch.zeros(
        #     self.lstm.num_layers * self.num_directions,
        #     batch_size,
        #     self.lstm.hidden_size
        # ), requires_grad=False)
        # 修改
        h0 = mindspore.Parameter(mindspore.ops.Zeros(
            self.lstm.num_layers * self.num_directions,
            batch_size,
            self.lstm.hidden_size
        ), requires_grad=False)
        c0 = mindspore.Parameter(mindspore.ops.Zeros(
            self.lstm.num_layers * self.num_directions,
            batch_size,
            self.lstm.hidden_size
        ), requires_grad=False)

        # if CUDA:
        #     return h0.cuda(), c0.cuda()
        # else:
        #     return h0, c0
        # 修改，删除-注释
        return h0, c0

    # def forward(self, src_embedding, srclens, srcmask, temp=1):
    # 修改 调试
    def construct(self, src_embedding, srclens, srcmask, temp=1):
        def init_state_2(self, input):
            # batch_size = input.size(0)  # retrieve dynamically for decoding
            # 修改
            # input = Tensor(input_x)
            # print(input.shape[0])
            # 输入的第一个维度是 batch_size 的值
            batch_size = input.shape[0] # retrieve dynamically for decoding
            # h0 = Variable(torch.zeros(
            #     self.lstm.num_layers * self.num_directions,
            #     batch_size,
            #     self.lstm.hidden_size
            # ), requires_grad=False)
            # c0 = Variable(torch.zeros(
            #     self.lstm.num_layers * self.num_directions,
            #     batch_size,
            #     self.lstm.hidden_size
            # ), requires_grad=False)
            # 修改
            h0 = mindspore.Parameter(mindspore.ops.Zeros(
                self.lstm.num_layers * self.num_directions,
                batch_size,
                self.lstm.hidden_size
            ), requires_grad=False)
            c0 = mindspore.Parameter(mindspore.ops.Zeros(
                self.lstm.num_layers * self.num_directions,
                batch_size,
                self.lstm.hidden_size
            ), requires_grad=False)

            # if CUDA:
            #     return h0.cuda(), c0.cuda()
            # else:
            #     return h0, c0
            # 修改，删除-注释
            return h0, c0
        h0, c0 = self.init_state(src_embedding)

        # h0, c0 = self.init_state(src_embedding)

        # if self.pack:
        #     inputs = pack_padded_sequence(src_embedding, srclens, batch_first=True)
        # else:
        #     inputs = src_embedding
        # 修改
        inputs = src_embedding

        outputs, (h_final, c_final) = self.lstm(inputs, (h0, c0))

        # if self.pack:
        #     outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # 修改，删除 上面2行



        return outputs, (h_final, c_final)
