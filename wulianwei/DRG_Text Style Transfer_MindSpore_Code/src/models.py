# 已修改部分


"""Sequence to Sequence models."""
import glob
import math

import numpy as np
import os

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import HeUniform, Uniform

import src.decoders as decoders
import src.encoders as encoders

# from src.cuda import CUDA


def get_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    # nothing to load, continue with fresh params
    if len(ckpts) == 0:
        return -1, None
    ckpts = map(lambda ckpt: (
        int(ckpt.split('.')[1]),
        ckpt), ckpts)
    # get most recent checkpoint
    epoch, ckpt_path = sorted(ckpts)[-1]
    return epoch, ckpt_path


def attempt_load_model(model, checkpoint_dir=None, checkpoint_path=None):
    assert checkpoint_dir or checkpoint_path

    if checkpoint_dir:
        epoch, checkpoint_path = get_latest_ckpt(checkpoint_dir)
    else:
        epoch = int(checkpoint_path.split('.')[-2])

    if checkpoint_path:
        # model.load_state_dict(torch.load(checkpoint_path))
        # 修改
        # net = model()
        # param_model = mindspore.load_param_into_net(net, mindspore.load_checkpoint(checkpoint_path))
        # print('Load from %s sucessful!' % checkpoint_path)

        model = model
        param_dict = mindspore.load_checkpoint("./working_dir/model.1.ckpt")
        param_not_load = mindspore.load_param_into_net(model, param_dict)
        print('Load from %s sucessful!' % checkpoint_path)


        return model, epoch + 1
    else:
        return model, 0


# class SeqModel(nn.Module):
# 修改
class SeqModel(nn.Cell):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        pad_id_src,
        pad_id_tgt,
        config=None,
    ):
        """Initialize model."""
        super(SeqModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_id_src = pad_id_src
        self.pad_id_tgt = pad_id_tgt
        self.batch_size = config['data']['batch_size']
        self.config = config
        self.options = config['model']
        self.model_type = config['model']['model_type']

        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.options['emb_dim'],
            # self.pad_id_src
            # 调试 2
            padding_idx=pad_id_src)

        if self.config['data']['share_vocab']:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(
                self.tgt_vocab_size,
                self.options['emb_dim'],
                # self.pad_id_tgt
                # 修改 2 调试
                padding_idx=pad_id_tgt)

        if self.options['encoder'] == 'lstm':
            self.encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'])
            # self.ctx_bridge = nn.Linear(
            # 修改
            self.ctx_bridge = nn.Dense(
                self.options['src_hidden_dim'],
                self.options['tgt_hidden_dim'])

        else:
            raise NotImplementedError('unknown encoder type')

        # # # # # #  # # # # # #  # # # # #  NEW STUFF FROM STD SEQ2SEQ
        
        if self.model_type == 'delete':
            # self.attribute_embedding = nn.Embedding(
            #     num_embeddings=2,
            #     embedding_dim=self.options['emb_dim'])
            # 修改
            self.attribute_embedding = nn.Embedding(
                vocab_size=2,
                embedding_size=self.options['emb_dim'])
            attr_size = self.options['emb_dim']

        elif self.model_type == 'delete_retrieve':
            self.attribute_encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'],
                pack=False)
            attr_size = self.options['src_hidden_dim']

        elif self.model_type == 'seq2seq':
            attr_size = 0

        else:
            raise NotImplementedError('unknown model type: %s. Accepted values: [seq2seq, delete_retrieve, delete]' % self.model_type)

        # self.c_bridge = nn.Linear(
        # 修改
        self.c_bridge = nn.Dense(
            attr_size + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])
        # self.h_bridge = nn.Linear(
        # 修改
        self.h_bridge = nn.Dense(
            attr_size + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])

        # # # # # #  # # # # # #  # # # # # END NEW STUFF

        self.decoder = decoders.StackedAttentionLSTM(config=config)

        # self.output_projection = nn.Linear(
        # 修改
        self.output_projection = nn.Dense(
            self.options['tgt_hidden_dim'],
            tgt_vocab_size)

        # self.softmax = nn.Softmax(dim=-1)
        # 修改
        self.softmax = mindspore.nn.Softmax(axis=-1)

        self.init_weights()



    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        # self.src_embedding.weight.data.uniform_(-initrange, initrange)
        # self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        # self.h_bridge.bias.data.fill_(0)
        # self.c_bridge.bias.data.fill_(0)
        # self.output_projection.bias.data.fill_(0)
        # # 修改 2 调试
        # self.flatten_weights()
        # # newwork = nn.Seq
        # # param_weight1 = self.src_embedding.parameters_and_names('weight')
        # # param_weight1 = self.get_parameters('weight')
        # param_weight1 = self.src_embedding.parameters_dict()['weight']
        # # parameters_dict()['weight']
        # param_weight2 = self.tgt_embedding.parameters_and_names('weight')
        # param_weight3 = self.h_bridge.parameters_and_names('bias')
        # param_weight4 = self.c_bridge.parameters_and_names('bias')
        # param_weight5 = self.output_projection.parameters_and_names('bias')
        # update1 = nn.ParameterUpdate(param_weight1)
        # update2 = nn.ParameterUpdate(param_weight2)
        # update3 = nn.ParameterUpdate(param_weight3)
        # update4 = nn.ParameterUpdate(param_weight4)
        # update5 = nn.ParameterUpdate(param_weight5)
        # weight1 = mindspore.common.initializer.Uniform(initrange)
        # weight2 = mindspore.common.initializer.Uniform(initrange)
        # weight3 = mindspore.ops.Fill(0)
        # weight4 = mindspore.ops.Fill(0)
        # weight5 = mindspore.ops.Fill(0)
        # self.set_data()
        # output1 = update1(weight1)
        # output2 = update2(weight2)
        # output3 = update3(weight3)
        # output4 = update4(weight4)
        # output5 = update5(weight5)

        # self.src_embedding.init_parameters_data()
        # self.src_embedding.
        # self.src_embedding.set_data_parallel()
    #     hiddendim_src = self.options['src_hidden_dim']
    #     hiddendim_tgt = self.options['tgt_hidden_dim']
    #     dropout = self.options['dropout']
    #
        weight = HeUniform(math.sqrt(5))
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1/math.sqrt(512 * 2))
        nn.Dense(128, 512, weight_init=weight_init, bias_init=bias_init)
        # nn.Dense(weight_init=weight1, bias_init=weight2)
    #     self.dropout = nn.Dropout(1-dropout)

    # def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask):
    # 修改
    def construct(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask):
        print("***********************construct函数***********************")
        src_emb = self.src_embedding(input_src)

        # srcmask = (1-srcmask).byte()
        # 修改 调试 ，可能有误
        srcmask = 1 - srcmask


        # src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, srclens, srcmask)
        # 修改 调试,下行代码需调整（源代码为上一行代码），考虑 将encoder、decoder 一起放到model文件中
        # 当前 调试 的位置
        src_outputs, (src_h_t, src_c_t) = self.encoder(src_embedding=src_emb, srclens=srclens, srcmask=srcmask)
        # src_outputs, (src_h_t, src_c_t) = encoders.LSTMEncoder.construct(self, src_embedding=src_emb, srclens=srclens, srcmask=srcmask)

        if self.options['bidirectional']:
            # h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            # c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
            # 修改 为 以下6行
            concat_op = ops.Concat(axis=1)
            cast_op = ops.Cast()
            h_t = concat_op((cast_op(src_h_t[-1], ms.float32), src_h_t[-2]))

            concat_op = ops.Concat(axis=1)
            cast_op = ops.Cast()
            c_t = concat_op((cast_op(src_c_t[-1], ms.float32), src_c_t[-2]))
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        src_outputs = self.ctx_bridge(src_outputs)


        # # # #  # # # #  # #  # # # # # # #  # # seq2seq diff
        # join attribute with h/c then bridge 'em
        # TODO -- put this stuff in a method, overlaps w/above

        if self.model_type == 'delete':
            # just do h i guess?
            a_ht = self.attribute_embedding(input_attr)
            # c_t = torch.cat((c_t, a_ht), -1)
            # h_t = torch.cat((h_t, a_ht), -1)
            # 修改 为 以下6行
            concat_op = ops.Concat(axis=-1)
            cast_op = ops.Cast()
            c_t = concat_op((cast_op(c_t, ms.float32), a_ht))

            concat_op = ops.Concat(axis=-1)
            cast_op = ops.Cast()
            h_t = concat_op((cast_op(h_t, ms.float32), a_ht))

        elif self.model_type == 'delete_retrieve':
            attr_emb = self.src_embedding(input_attr)
            _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attrlens, attrmask)
            if self.options['bidirectional']:
                # a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
                # a_ct = torch.cat((a_ct[-1], a_ct[-2]), 1)
                # 修改 为 以下6行
                concat_op = ops.Concat(axis=1)
                cast_op = ops.Cast()
                a_ht = concat_op((cast_op(a_ht[-1], ms.float32), a_ht[-2]))

                concat_op = ops.Concat(axis=1)
                cast_op = ops.Cast()
                a_ct = concat_op((cast_op(a_ct[-1], ms.float32), a_ct[-2]))
            else:
                a_ht = a_ht[-1]
                a_ct = a_ct[-1]

            # h_t = torch.cat((h_t, a_ht), -1)
            # c_t = torch.cat((c_t, a_ct), -1)
            # 修改 为 以下6行
            concat_op = ops.Concat(axis=-1)
            cast_op = ops.Cast()
            h_t = concat_op((cast_op(h_t, ms.float32), a_ht))

            concat_op = ops.Concat(axis=-1)
            cast_op = ops.Cast()
            c_t = concat_op((cast_op(c_t, ms.float32), a_ct))

            
        c_t = self.c_bridge(c_t)
        h_t = self.h_bridge(h_t)

        # # # #  # # # #  # #  # # # # # # #  # # end diff

        tgt_emb = self.tgt_embedding(input_tgt)
        tgt_outputs, (_, _) = self.decoder(
            tgt_emb,
            (h_t, c_t),
            src_outputs,
            srcmask)

        tgt_outputs_reshape = tgt_outputs.contiguous().view(
            tgt_outputs.size()[0] * tgt_outputs.size()[1],
            tgt_outputs.size()[2])
        decoder_logit = self.output_projection(tgt_outputs_reshape)
        decoder_logit = decoder_logit.view(
            tgt_outputs.size()[0],
            tgt_outputs.size()[1],
            decoder_logit.size()[1])

        probs = self.softmax(decoder_logit)

        return decoder_logit, probs

    def count_params(self):
        n_params = 0
        # for param in self.parameters():
        # 修改
        for param in self.get_parameters():
            # n_params += np.prod(param.data.cpu().numpy().shape)
            # 修改
            n_params += np.prod(param.data.shape)
        return n_params
        
        
        
