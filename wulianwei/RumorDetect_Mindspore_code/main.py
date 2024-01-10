from scipy.fftpack import fft, dct
import os
from PIL import Image
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore import context
import mindspore.dataset as ds
import mindspore_resnet50 as ms_resnet


class multimodal_attention(nn.Cell):
    """
    基础的 Attention 模块
    """
    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        # self.softmax = ops.Softmax(axis=2)
        self.softmax = nn.Softmax(axis=2)

    def construct(self, q, k, v, scale=None, attn_mask=None):
        attention = ops.matmul(q, k.transpose(-2, -1))
        # print('attention.shape:{}'.format(attention.shape))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = ops.masked_fill(attention, attn_mask, -np.inf)  # 掩码全部变为 -np.inf
        attention = self.softmax(attention)
        # print('attention.softmax:{}'.format(attention))
        attention = self.dropout(attention)
        attention = ops.matmul(attention, v)
        # print('attn_final.shape:{}'.format(attention.shape))
        return attention


class MultiHeadAttention(nn.Cell):
    """
    多头注意力层
    """
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Dense(1, self.dim_per_head * num_heads, has_bias=False)
        self.linear_v = nn.Dense(1, self.dim_per_head * num_heads, has_bias=False)
        self.linear_q = nn.Dense(1, self.dim_per_head * num_heads, has_bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Dense(model_dim, 1, has_bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([model_dim])

    def construct(self, query, key, value, attn_mask=None):
        residual = query
        query = ops.expand_dims(query, -1)
        key = ops.expand_dims(key, -1)
        value = ops.expand_dims(value, -1)
        # print("query.shape:{}".format(query.shape))

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        # print('key.shape:{}'.format(key.shape))

        # split by heads
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5

        attention = self.dot_product_attention(query, key, value,
                                               scale, attn_mask)

        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)
        # print('attention_con_shape:{}'.format(attention.shape))

        # final linear projection
        output = ops.squeeze(self.linear_final(attention), -1)
        # print('output.shape:{}'.format(output.shape))
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class PositionalWiseFeedForward(nn.Cell):
    """
    全连接层
    """
    def __init__(self, model_dim=256, ffn_dim=2048, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Dense(model_dim, ffn_dim)
        self.w2 = nn.Dense(ffn_dim, model_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([model_dim])
        self.relu = nn.ReLU()

    def construct(self, x):
        residual = x
        x = self.w2(self.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        output = x
        return output


def ConvBNRelu2d(in_channels, out_channels, kernel_size, stride=1):
    # print(in_channels, out_channels, kernel_size, stride, padding)
    if kernel_size == 3:
        result_layer = nn.SequentialCell(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=stride,
            ),
            nn.MaxPool2d(
                kernel_size=(1, kernel_size),
                stride=stride,
                pad_mode='same'
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()  # torch 版本在这里是允许就地操作
        )
    else:  # kernel_size = 1
        result_layer = nn.SequentialCell(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=stride,
            ),
            nn.MaxPool2d(
                kernel_size=(1, kernel_size),
                stride=stride,
                pad_mode='same'
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()  # torch 版本在这里是允许就地操作
        )
    return result_layer


class DctStem(nn.Cell):  # dct 主干
    def __init__(self, kernel_sizes, num_channels):
        super(DctStem, self).__init__()
        self.conv_layer = nn.SequentialCell(
            ConvBNRelu2d(
                in_channels=1,
                out_channels=num_channels[0],
                kernel_size=kernel_sizes[0]
            ),
            ConvBNRelu2d(
                in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=kernel_sizes[1]
            ),
            ConvBNRelu2d(
                in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=kernel_sizes[2]
            ),
            nn.MaxPool2d(kernel_size=(1, 2), pad_mode='same')
        )

    def construct(self, dct_img):  # [4, 64, 250]
        tmp_img_1 = ops.expand_dims(dct_img, axis=1)  # 在 1 维轴上拓展 1 维 [4, 1, 64, 250]
        tmp_img_2 = self.conv_layer(tmp_img_1)  # tmp_img_2.ahep →
        result_img = ops.transpose(tmp_img_2, (0, 2, 1, 3))  # [4, 64, 1, 250]
        return result_img


class DctInceptionBlock(nn.Cell):
    def __init__(
            self,
            in_channel=128,
            branch1_channels=None,
            branch2_channels=None,
            branch3_channels=None,
            branch4_channels=None,
    ):
        super(DctInceptionBlock, self).__init__()
        if branch1_channels is None:
            branch1_channels = [64]
        if branch2_channels is None:
            branch2_channels = [48, 64]
        if branch3_channels is None:
            branch3_channels = [64, 96, 96]
        if branch4_channels is None:
            branch4_channels = [32]
        self.branch_1 = ConvBNRelu2d(
            in_channels=in_channel,
            out_channels=branch1_channels[0],
            kernel_size=1
        )
        self.branch_2 = nn.SequentialCell(
            ConvBNRelu2d(
                in_channels=in_channel,
                out_channels=branch2_channels[0],
                kernel_size=1
            ),
            ConvBNRelu2d(
                in_channels=branch2_channels[0],
                out_channels=branch2_channels[1],
                kernel_size=3,
                # padding=0  # 源 torch 是 padding=(0, 1), 但有 warning
            )
        )
        self.branch_3 = nn.SequentialCell(
            ConvBNRelu2d(
                in_channels=in_channel,
                out_channels=branch3_channels[0],
                kernel_size=1
            ),
            ConvBNRelu2d(
                in_channels=branch3_channels[0],
                out_channels=branch3_channels[1],
                kernel_size=3,
                # padding=0  # 源 torch 是 padding=(0, 1), 但有 warning
            ),
            ConvBNRelu2d(
                in_channels=branch3_channels[1],
                out_channels=branch3_channels[2],
                kernel_size=3,
                # padding=0  # 源 torch 是 padding=(0, 1), 但有 warning
            ),
        )
        self.branch_4 = nn.SequentialCell(
            nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=1,
                pad_mode='same'  # 源 torch 是 padding=(0, 1), 但有 warning
            ),
            ConvBNRelu2d(
                in_channels=in_channel,
                out_channels=branch4_channels[0],
                kernel_size=1
            ),
        )

    def construct(self, img):
        tmp_img = ops.transpose(img, (0, 2, 1, 3))
        out_1 = self.branch_1(tmp_img)  # [4, 1, 64, 250]
        out_2 = self.branch_2(tmp_img)
        out_3 = self.branch_3(tmp_img)
        out_4 = self.branch_4(tmp_img)
        output = ops.concat((out_1, out_2, out_3, out_4), axis=1)
        output = ops.transpose(output, (0, 2, 1, 3))
        return output


class DctCNN(nn.Cell):
    def __init__(self,
                 dropout,
                 kernel_sizes,
                 num_channels,  # [32, 64, 128]
                 in_channel=None,
                 branch1_channels=None,
                 branch2_channels=None,
                 branch3_channels=None,
                 branch4_channels=None,
                 out_channels=64):
        super(DctCNN, self).__init__()
        if in_channel is None:
            in_channel = 128
        if branch4_channels is None:
            branch4_channels = [32]
        if branch3_channels is None:
            branch3_channels = [64, 96, 96]
        if branch2_channels is None:
            branch2_channels = [48, 64]
        if branch1_channels is None:
            branch1_channels = [64]
        self.dctStem = DctStem(
            kernel_sizes=kernel_sizes,
            num_channels=num_channels
        )
        self.dctInceptionBlock = DctInceptionBlock(
            in_channel=in_channel,
            branch1_channels=branch1_channels,
            branch2_channels=branch2_channels,
            branch3_channels=branch3_channels,
            branch4_channels=branch4_channels,
        )
        self.maxPool = nn.MaxPool2d((1, 122), pad_mode='same')
        self.dropout = nn.Dropout(dropout)
        self.conv = ConvBNRelu2d(
            branch1_channels[-1] + branch2_channels[-1] +
            branch3_channels[-1] + branch4_channels[-1],
            out_channels,
            kernel_size=1
        )

    def construct(self, dct_img):
        dct_f = self.dctStem(dct_img)  # dct_f 为频域特征 [4, 1, 64, 250]
        output = self.dctInceptionBlock(dct_f)
        output = self.maxPool(output)
        t1 = ops.transpose(output, (0, 2, 1, 3))
        output = self.conv(t1)
        t2 = ops.transpose(output, (0, 2, 1, 3))  # 得到一个 4 64 64 250 的矩阵, 只要 4 64 64
        tmp_out = t2.resize(4, 64, 64)
        tmp_out = ops.reshape(tmp_out, (-1, 4096))
        # output = tmp_out.reshape(-1, 4096)  # 得到一个 1000 4096
        return tmp_out


class multimodal_fusion_layer(nn.Cell):
    """
    多模态融合层
    """
    def __init__(self, model_dim=256, num_heads=8, ffn_dim=2048, dropout=0.5):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

        self.fusion_linear = nn.Dense(model_dim * 2, model_dim)

    def construct(self, image_output, dct_output, attn_mask=None):
        output_1 = self.attention_1(image_output, dct_output, dct_output,
                                    attn_mask)
        output_2 = self.attention_2(dct_output, image_output, image_output,
                                    attn_mask)
        # print('attention out_shape:{}'.format(output.shape))
        output1 = self.feed_forward_1(output_1)
        output2 = self.feed_forward_2(output_2)
        output = ops.concat((output1, output2), axis=1)
        res = self.fusion_linear(output)
        return res


class NetShareFusion(nn.Cell):
    """
    整个网络
    """
    def __init__(self,
                 pth_file,
                 kernel_sizes,
                 num_channels,
                 model_dim,
                 drop_and_BN,
                 num_labels=2,
                 # num_layers=1,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.5):
        super(NetShareFusion, self).__init__()
        self.model_dim = model_dim
        self.pth_file = pth_file
        self.drop_and_BN = drop_and_BN
        self.dropout = nn.Dropout(dropout)

        # image
        self.resnet50 = ms_resnet.resnet18(model_dim)
        tmp_result = count_params(self.resnet50)
        print('resnet50 的参数量:', tmp_result)
        self.linear_image = nn.Dense(1000, model_dim)
        self.bn_resnet50 = nn.BatchNorm1d(model_dim)

        # dct_image
        self.dct_img = DctCNN(
            dropout,
            kernel_sizes,
            num_channels,
            in_channel=128,
            branch1_channels=[64],
            branch2_channels=[48, 64],
            branch3_channels=[64, 96, 96],
            branch4_channels=[32],
            out_channels=64
        )
        self.linear_dct = nn.Dense(4096, model_dim)
        self.bn_dct = nn.BatchNorm1d(model_dim)

        # multimodal fusion (原本有两层)
        self.fusion_layers = multimodal_fusion_layer(model_dim, num_heads, ffn_dim, dropout)

        # classifier
        self.linear_1 = nn.Dense(model_dim, 35)
        self.bn_1 = nn.BatchNorm1d(35)
        self.linear_2 = nn.Dense(35, num_labels)
        self.softmax = nn.Softmax(axis=1)
        self.relu = nn.ReLU()

    def drop_BN_layer(self, x, part):
        if part == 'dct':
            x = self.dropout(x)
            x = self.bn_dct(x)
        elif part == 'resnet50':
            x = self.dropout(x)
            x = self.bn_resnet50(x)
        else:
            pass
            # print('drop_BN_layer 出错')
            # exit(0)
        return x

    def construct(self, image, dct_img, attn_mask):
        # visual feature
        img_out = self.resnet50(image)
        img_out = self.linear_image(img_out)
        img_out = self.relu(img_out)
        img_out = self.drop_BN_layer(img_out, part='resnet50')

        # dct feature
        dct_out = self.dct_img(dct_img)
        dct_out = self.relu(self.linear_dct(dct_out))
        dct_out = self.drop_BN_layer(dct_out, part='dct')
        return dct_out
        # # for fusion_layer in self.fusion_layers:
        # img_out = self.fusion_layers(img_out, dct_out, attn_mask)
        # output = self.relu(self.linear_1(img_out))
        # output = self.dropout(output)
        # output = self.linear_2(output)
        # y_prediction_prob = self.softmax(output)
        # return img_out, y_prediction_prob


def process_dct_img(img):
    img = np.array(img[0])  # size = [1, 224, 224]
    # print('之后 process_dct_img img:', img, img.shape)
    height = img.shape[1]  # height:224
    width = img.shape[2]
    # print('height:{}'.format(height))
    N = 8
    step = int(height / N)  # 28
    dct_img = np.zeros((1, N * N, step * step, 1), dtype=np.float32)  # [1,64,784,1]
    fft_img = np.zeros((1, N * N, step * step, 1))  # [1,64,784,1]
    # print('dct_img:{}'.format(dct_img.shape))
    # print('fft_img:{}'.format(fft_img.shape))

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row + step), col:(col + step)], dtype=np.float32)
            # print('block:{}'.format(block.shape))
            block1 = block.reshape(-1, step * step, 1)  # [batch_size,784,1]
            # print('block1:{}'.format(block1.shape))
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]
            i += 1

    # for i in range(64):
    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real  # [batch_size,64, 784,1]
    # print('1:', fft_img.shape, type(fft_img))
    fft_img = ms.Tensor.from_numpy(fft_img)  # [batch_size, 64, 784, 1]
    fft_img = ms.Tensor(fft_img, ms.float32)
    # print('2:', fft_img.shape, type(fft_img))
    new_img = ops.interpolate(fft_img, None, None, sizes=(250, 1), mode="bilinear")  # [batch_size, 64, 250, 1]
    # print('3', new_img.shape, type(new_img))
    new_img = ops.squeeze(new_img, axis=0)
    # print('4', new_img.shape, type(new_img))
    new_img = ops.squeeze(new_img, axis=-1)  # torch.size = [64, 250]
    # print('5', new_img.shape, type(new_img))

    return new_img


def read_images(file_List):
    image_List = {}
    img_Num = 0
    for path in file_List:
        for filename in os.listdir(path):
            img = Image.open(path + filename).convert('RGB')
            img_id = filename.split('.')[0]
            image_List[img_id] = img
            if img_Num % 500 == 0:
                print('图像读取完成数量:', img_Num)
            img_Num += 1
    return image_List, img_Num


def select_image(image_Num, image_id_list, image_List):
    for i in range(image_Num):
        image_id = image_id_list[i]
        sta = image_id.find('large/')
        end = image_id.find('.jpg')
        true_image_id = image_id[sta + 6:end]
        # print('true_image_id:', true_image_id)
        if true_image_id in image_List:
            # print('Yes, img_id:{}'.format(true_image_id))
            return true_image_id
    return False


def get_data(dataset_type, image_List):
    if dataset_type == 'train':
        data_file = './tweets/train_non-rumor.txt'
    elif dataset_type == 'test':
        data_file = './tweets/test_non-rumor.txt'
    else:
        data_file = './'
        print('get_data() 出错')
        exit(1)

    file = open(data_file, 'r', encoding='UTF-8')
    lines = file.readlines()
    data_post_id = []
    data_post_content = []
    data_image = []
    data_label = []

    data_num = len(lines)
    print('总共的新闻数量 data_num:', data_num // 3)
    unmatched_num = 0

    # 每次读三行
    i = 0
    while i < len(lines):
        tmp_line = lines[i].split('|')
        post_id = tmp_line[0]
        # print('post_id:', post_id)
        label = 0  # tmp_line[5]
        i = i + 1
        tmp_line = lines[i].split('|')
        image_id_list = tmp_line[:-1]
        # print('image_id_list:', image_id_list)
        img_Num = len(image_id_list)
        # print('img_Num:', img_Num)
        i = i + 1
        post_content = lines[i].strip()
        # print('post_content:', post_content)
        image_id = select_image(img_Num, image_id_list, image_List)  # 只选一张照片出来
        if image_id is not False:
            image = image_List[image_id]
            data_post_id.append(int(post_id))
            data_post_content.append(post_content)
            data_image.append(image)
            data_label.append(int(label))
        else:
            unmatched_num += 1
        i = i + 1
    file.close()
    data_dic = {'post_id': np.array(data_post_id),
                'post_content': data_post_content,
                'image': data_image,
                'label': np.array(data_label)
                }
    return data_dic, data_num // 3 - unmatched_num


class MyDataset(object):
    def __init__(self, my_data, transforms_res50=None, transform_dct=None):
        super(MyDataset, self).__init__()
        self.transforms_resnet50 = transforms_res50
        self.transform_dct = transform_dct
        self.post_id = ms.Tensor.from_numpy(my_data['post_id'])
        self.image = list(my_data['image'])
        self.label = ms.Tensor.from_numpy(my_data['label'])
        self.label = ms.Tensor(self.label, ms.int64)

    def __getitem__(self, idx):
        idx = int(idx)  # 不改的话就是 numpy.int, 而 ms.Tensor 无法读取这样的下标
        # print('1:', self.image[idx], type(self.image[idx]))
        # print('2:', self.image[idx].convert('L'), type(self.image[idx].convert('L')))
        dct_img = self.transform_dct(self.image[idx].convert('L'))
        # print('3:', dct_img[0].shape, type(dct_img[0]))
        dct_img = process_dct_img(dct_img)
        # print('4:', dct_img, dct_img.shape, type(dct_img))
        item_dct_img = dct_img.asnumpy()
        # print('4 1:', item_dct_img, type(item_dct_img))
        item_image = self.transforms_resnet50(self.image[idx])[0]
        # print('5:', item_image, item_image.shape, type(item_image))
        item_post_id = self.post_id[idx]
        # print('6:', item_post_id, type(item_post_id))
        item_label = self.label[idx]
        return item_image, item_dct_img, item_post_id, item_label

    def __len__(self):
        return len(self.label)


def count_params(net):
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params


def handle_batch_input(trainData):
    vgg_paras = ["image"]
    dct_paras = ["dct_img"]
    share_paras = ['label', 'post_id']
    parameters = {}
    involve = vgg_paras + dct_paras
    involve += share_paras

    for para in involve:
        parameters[para] = trainData[para]

    return parameters


# def train_loop(net, dataSet, loss_fn, opt):
#     # Define forward function
#     def forward_fn(input_data, input_label):
#         # print("input_data:", input_data)
#         # def construct(self, image, dct_img, attn_mask):
#         logit = net(image=input_data['image'], dct_img=input_data['dct_img'], attn_mask=None)
#         tmp_loss = loss_fn(logit, input_label)
#         return tmp_loss
#
#     # Get gradient function
#     grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)
#     # print('grad_fn:', grad_fn)
#
#     # Define function of one-step training
#     def train_step(input_data, input_label):
#         tmp_result = grad_fn(input_data, input_label)
#         (tmp_loss, _), grads = tmp_result
#         # print('tmp_result:', tmp_result)
#         tmp_loss = ops.depend(tmp_loss, opt(grads))
#         return tmp_loss
#
#     size = dataSet.get_dataset_size()
#     # print('size:', size)
#     net.set_train(False)
#     current = 0
#     for all_data in dataSet.create_dict_iterator():
#         # print('all_data:', all_data)
#         # label = all_data['label']
#         # image = all_data['image']
#         # dct_img = all_data['dct_img']
#         # attn_mask = None
#         parameters = handle_batch_input(all_data)
#         # print("parameters:", parameters)
#         logit = net(image=parameters['image'], dct_img=parameters['dct_img'], attn_mask=None)
#         print('logit', logit)
#         # loss = train_step(parameters, label)
#         # loss = loss.asnumpy()
#         # current = current + 1
#         print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
#         exit(0)


# def wly_test_loop(model, data_set, loss_func):
#     num_batches = data_set.get_dataset_size()
#     print('num_batches:', num_batches)
#     model.set_train(False)
#     total, test_loss, correct = 0, 0, 0
#     for data in data_set.create_dict_iterator():
#         print('len(data):', len(data))
#         pred = model(data[0], data[1], None)
#         total += len(data)
#         test_loss += loss_func(pred, data[3]).asnumpy()
#         correct += (pred.argmax(1) == data[3]).asnumpy().sum()
#     test_loss /= num_batches
#     correct /= total
#     print(f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    if context.get_context('device_target') == 'GPU':
        context.set_context(device_target="GPU")  # 如果有 gpu 就用 gpu 吧
    print("设备环境:", context.get_context('device_target'))

    image_file_list = ['./rumor_images_wly/', './non-rumor_images_wly/']
    image_list, img_num = read_images(image_file_list)
    print('新闻文本条数 len(image_list):', img_num, type(image_list))

    train_data, train_data_num = get_data('train', image_list)
    print('匹配的训练集样本数 train_data_num:', train_data_num, type(train_data))
    transforms_dct = transforms.Compose([
        vision.Resize((224, 224)),
        vision.ToTensor()
    ])
    transforms_resnet50 = transforms.Compose([
        vision.Resize((224, 224)),
        vision.ToTensor(),
    ])
    train_dataset = MyDataset(my_data=train_data,
                              transforms_res50=transforms_resnet50,
                              transform_dct=transforms_dct)
    # print('1 train_dataset:', train_dataset, type(train_dataset))
    dataset = ds.GeneratorDataset(source=train_dataset, column_names=["image", "dct_img", "post_id", "label"])
    dataset = dataset.batch(batch_size=4)
    # dataLoader = dataset.create_dict_iterator()
    network = NetShareFusion(pth_file='',
                             kernel_sizes=[3, 3, 3],
                             num_channels=[32, 64, 128],
                             model_dim=1000,
                             # num_layers=2,
                             num_heads=4,
                             drop_and_BN='drop-BN',
                             dropout=0.5)
    # print('network:', network)
    # print("resnet_50:", resnet_50)
    result = count_params(network)
    print('总的参数量:', result)
    cnt = 0
    optimizer = nn.Adam(network.trainable_params(), learning_rate=1e-4)
    # criterion = nn.SoftmaxCrossEntropyWithLogits()
    # model = ms.Model(network, criterion, optimizer, metrics={"Accuracy": nn.Accuracy()})
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = nn.SGD(network.trainable_params(), learning_rate=learning_rate)
    epochs = 2
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # train_loop(model, train_dataset, loss_fn, optimizer)
        # wly_test_loop(network, dataset, loss_fn)
        num_batches = dataset.get_dataset_size()
        print('num_batches:', num_batches)
        network.set_train(False)
        total, test_loss, correct = 0, 0, 0
        for data in dataset.create_dict_iterator():
            print(data['image'].shape, data['dct_img'].shape, data['post_id'].shape, data['label'].shape)
            pred = network(data['image'], data['dct_img'], None)
            print('pred:', pred, pred.shape)
            exit(0)
        #     total += len(data)
        #     test_loss += loss_fn(pred[1], data['label']).asnumpy()
        #     correct += (pred.argmax(1) == data[3]).asnumpy().sum()
        # test_loss /= num_batches
        # correct /= total
        # print(f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("Done!")
