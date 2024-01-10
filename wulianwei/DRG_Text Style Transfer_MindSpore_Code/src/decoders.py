import math
import numpy as np

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops




# class BilinearAttention(nn.Module):
# 修改
class BilinearAttention(nn.Cell):
    """ bilinear attention layer: score(H_j, q) = H_j^T W_a q
                (where W_a = self.in_projection)
    """
    def __init__(self, hidden):
        super(BilinearAttention, self).__init__()
        # self.in_projection = nn.Linear(hidden, hidden, bias=False)
        # self.softmax = nn.Softmax()
        # self.out_projection = nn.Linear(hidden * 2, hidden, bias=False)
        # self.tanh = nn.Tanh()
        # 修改
        self.in_projection = nn.Dense(hidden, hidden, bias=False)
        self.softmax = 	mindspore.nn.SoftmaxCrossEntropyWithLogits()
        self.out_projection = nn.Dense(hidden * 2, hidden, bias=False)
        self.tanh = mindspore.nn.Tanh()

    def forward(self, query, keys, srcmask=None, values=None):
        """
            query: [batch, hidden]
            keys: [batch, len, hidden]
            values: [batch, len, hidden] (optional, if none will = keys)

            compare query to keys, use the scores to do weighted sum of values
            if no value is specified, then values = keys
        """
        if values is None:
            values = keys
    
        # [Batch, Hidden, 1]
        decoder_hidden = self.in_projection(query).unsqueeze(2)
        # [Batch, Source length]
        # attn_scores = torch.bmm(keys, decoder_hidden).squeeze(2)
        # 修改
        attn_scores = mindspore.ops.BatchMatMul(keys, decoder_hidden).squeeze(2)
        if srcmask is not None:
            attn_scores = attn_scores.masked_fill(srcmask, -float('inf'))
            
        attn_probs = self.softmax(attn_scores)
        # [Batch, 1, source length]
        attn_probs_transposed = attn_probs.unsqueeze(1)
        # [Batch, hidden]

        # weighted_context = torch.bmm(attn_probs_transposed, values).squeeze(1)
        # context_query_mixed = torch.cat((weighted_context, query), 1)
        # 修改
        weighted_context = mindspore.ops.BatchMatMul(attn_probs_transposed, values).squeeze(1)
        concat_op = ops.Concat(axis=1)
        cast_op = ops.Cast()
        context_query_mixed = concat_op((cast_op(weighted_context, ms.float32), query))


        context_query_mixed = self.tanh(self.out_projection(context_query_mixed))

        return weighted_context, context_query_mixed, attn_probs

# class AttentionalLSTM(nn.Module):
# 修改
class AttentionalLSTM(nn.Cell):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_dim, hidden_dim, config, attention):
        """Initialize params."""
        super(AttentionalLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.use_attention = attention
        self.config = config
        # self.cell = nn.LSTMCell(input_dim, hidden_dim)
        # 修改
        self.cell = mindspore.nn.LSTMCell(input_dim, hidden_dim)

        if self.use_attention:
            self.attention_layer = BilinearAttention(hidden_dim)

    # def forward(self, input, hidden, ctx, srcmask, kb=None):
    # 修改
    def construct(self, input, hidden, ctx, srcmask, kb=None):
        input = input.transpose(0, 1)

        output = []
        timesteps = range(input.size(0))
        for i in timesteps:
            hy, cy = self.cell(input[i], hidden)
            if self.use_attention:
                _, h_tilde, alpha = self.attention_layer(hy, ctx, srcmask)
                hidden = h_tilde, cy
                output.append(h_tilde)
            else: 
                hidden = hy, cy
                output.append(hy)

        # combine outputs, and get into [time, batch, dim]
        # output = torch.cat(output, 0).view(
        #     input.size(0), *output[0].size())
        # 修改
        concat_op = ops.Concat(axis=0)
        cast_op = ops.Cast()
        output = concat_op((cast_op(output, ms.float32))).view(
            input.size(0), *output[0].size())

        output = output.transpose(0, 1)

        return output, hidden

# class StackedAttentionLSTM(nn.Module):
# 修改
class StackedAttentionLSTM(nn.Cell):
    """ stacked lstm with input feeding
    """
    def __init__(self, cell_class=AttentionalLSTM, config=None):
        super(StackedAttentionLSTM, self).__init__()
        self.options=config['model']


        # self.dropout = nn.Dropout(self.options['dropout'])
        # 修改
        self.dropout = nn.Dropout(1-self.options['dropout'])

        self.layers = []
        input_dim = self.options['emb_dim']
        hidden_dim = self.options['tgt_hidden_dim']
        for i in range(self.options['tgt_layers']):
            layer = cell_class(input_dim, hidden_dim, config, config['model']['attention'])
            # self.add_module('layer_%d' % i, layer)
            # 修改，2 调试
            self.insert_child_to_cell('layer_%d' % i, layer)
            self.layers.append(layer)
            input_dim = hidden_dim


    def forward(self, input, hidden, ctx, srcmask, kb=None):
        h_final, c_final = [], []
        for i, layer in enumerate(self.layers):
            output, (h_final_i, c_final_i) = layer(input, hidden, ctx, srcmask, kb)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_final.append(h_final_i)
            c_final.append(c_final_i)

        # h_final = torch.stack(h_final)
        # c_final = torch.stack(c_final)
        # 修改
        h_final = mindspore.ops.Stack(h_final)
        c_final = mindspore.ops.Stack(c_final)

        return input, (h_final, c_final)


