from mindspore import nn, ops
import model
import mindspore as ms
from . import register_arc


@register_arc
class Resnet50(nn.Cell):
    def __init__(self, opt, weight_file_path):
        super(Resnet50, self).__init__()

        temp_net = model.resnet.ResNet.get_resnet50(num_classes=1000)

        params_dict = ms.load_checkpoint(weight_file_path)
        ms.load_param_into_net(temp_net, params_dict)

        self.resnet_model = temp_net

        self.resnet_model.fc = nn.Dense(self.resnet_model.fc.in_channels, opt.embed_dim)

        self.layer_blocks = nn.SequentialCell([self.resnet_model.layer1, self.resnet_model.layer2,
                                               self.resnet_model.layer3, self.resnet_model.layer4])

        self.out_adjust = None

    def construct(self, x):
        x = self.resnet_model.max_pool(self.resnet_model.relu(self.resnet_model.norm(self.resnet_model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        no_avg_feat = x  # [b, c, h, w]
        x = self.resnet_model.avg_pool(x)  # [b, c, 1, 1]
        enc_out = x = x.view(x.shape[0], -1)  # [b, c]

        x = self.resnet_model.fc(x)  # [b, embed_dim]

        x = ops.L2Normalize(axis=-1)(x)  # [b, embed_dim]

        # if 'normalize' in self.pars.arch:
        #     x = torch.nn.functional.normalize(x, dim=-1)  # [b, embed_dim]
        # if self.out_adjust and not self.train:
        #     x = self.out_adjust(x)

        # x: [b, embed_dim]; enc_out: [b, c]; no_avg_feat: [b, c, h, w]
        return x, (enc_out, no_avg_feat)
