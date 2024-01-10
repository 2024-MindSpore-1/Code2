from .others import *
import numpy as np


class ResNet(nn.Cell):

    def __init__(self, block, layers, rgbd, bbox, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, pad_mode="same")
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

        # Top layer
        # 用于conv5,因为没有更上一层的特征了，也不需要smooth的部分
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1)  # Reduce channels

        # Smooth layers
        # 分别用于conv4,conv3,conv2（按顺序）
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="same")
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="same")
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="same")

        # Lateral layers
        # 分别用于conv4,conv3,conv2（按顺序）
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.rgbd = rgbd
        # 1102
        self.yolobox = bbox
        # # 1026
        # self.adaAvgPool = nn.AdaptiveAvgPool2d((8, 8))
        # # 1102
        # self.avgpool_rgbonly = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc3 = nn.Linear(1024, 1024)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return ops.upsample(x, size=(H, W), mode='bilinear') + y

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.SequentialCell(*layers)

    # 20211102 rgb-only+yolobbox
    def _forward_impl_bbox(self, x, bbox):
        # Bottom-up  FPN
        c1 = ops.relu(self.bn1(self.conv1(x)))
        c1 = ops.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)  # torch.Size([1, 2048, 8, 8]) when image input ==(256,256)
        # pdb.set_trace()
        # c5 = self.adaAvgPool(c5) #lmj 1026 使不同尺寸的输入图片的输出相同->但这样会使小图片放大对营养评估是否有影响未知

        # before 1108
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)  # [b, 256, 67, 89]
        # (267, 356) ->p2(67,89)  需要设置对照组，以下代码当没有bbox时靠p2输出也要做一次
        # pdb.set_trace()
        # 怎么对batch中的每个特征图进行区域选择？？？？？？？？？？

        # 1108
        # H,W = 67,89
        # p2 = self.smooth1(F.upsample(self.toplayer(c5), size=(H,W), mode='bilinear'))

        output = []
        for i, box in enumerate(bbox):
            if box != '':  # 有几张图片没有bbox
                # pdb.set_trace()
                with open(box, "r+", encoding="utf-8", errors="ignore") as f:
                    # w,h = 89, 67   #resize后的图片
                    w, h = p2.shape[3], p2.shape[2]
                    allLabels = []
                    for line in f:
                        label = []
                        aa = line.split(" ")
                        # pdb.set_trace()
                        x_center = w * float(aa[1])  # aa[1]左上点的x坐标
                        y_center = h * float(aa[2])  # aa[2]左上点的y坐标
                        width = int(w * float(aa[3]))  # aa[3]图片width
                        height = int(h * float(aa[4]))  # aa[4]图片height
                        lefttopx = int(x_center - width / 2.0)
                        lefttopy = int(y_center - height / 2.0)
                        label = [lefttopx, lefttopy, lefttopx + width, lefttopy + height]
                        allLabels.append(label)

                    nparray = np.array(allLabels)
                    # 可能存在多个位置labels
                    lefttopx = nparray[:, 0].min()
                    lefttopy = nparray[:, 1].min()
                    # width = nparray[:,2].max()
                    # height = nparray[:,3].max()
                    left_plus_width = nparray[:, 2].max()
                    top_plus_height = nparray[:, 3].max()

                    # pdb.set_trace()
                    roi = p2[i][..., lefttopy + 1:top_plus_height + 3, lefttopx + 1:left_plus_width + 1]
                    # 池化统一大小
                    output.append(ops.adaptive_avg_pool2d(roi, (2, 2)))
            elif box == '':
                # pdb.set_trace()
                output.append(ops.adaptive_avg_pool2d(p2[i], (2, 2)))
        output = ops.stack(output, axis=0)
        x = ops.flatten(output, 1)
        x = self.fc3(x)
        x = ops.relu(x)
        results = []
        results.append(self.calorie(x).squeeze())  # 2048
        results.append(self.mass(x).squeeze())
        results.append(self.fat(x).squeeze())
        results.append(self.carb(x).squeeze())
        results.append(self.protein(x).squeeze())
        return results

    # Normal
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if not self.rgbd:
            # pdb.set_trace()
            # torch.Size([32, 3, 256, 256])#->torch.Size([32, 64, 128, 128])
            x = self.conv1(x)  # torch.Size([16, 3, 267, 356])
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)  # torch.Size([16, 2048, 9, 12])

            # pdb.set_trace()
            x = self.avgpool(x)  # 统一进行自适应平均池化，即使输入图片大小不同，x的输出也相同
            x = ops.flatten(x, 1)
            # x = self.fc(x)
            x = self.fc1(x)
            # 0721
            # x = self.dropout(x)
            x = self.fc2(x)
            # 0722
            # x = self.dropout(x)
            # pdb.set_trace()
            x = ops.relu(x)
            results = []
            results.append(self.calorie(x).squeeze())  # 2048
            results.append(self.mass(x).squeeze())
            results.append(self.fat(x).squeeze())
            results.append(self.carb(x).squeeze())
            results.append(self.protein(x).squeeze())
            return results

        elif self.rgbd:
            # Bottom-up  FPN
            c1 = ops.relu(self.bn1(self.conv1(x)))
            c1 = ops.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
            c2 = self.layer1(c1)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)  # torch.Size([1, 2048, 8, 8]) when image input ==(256,256)
            # pdb.set_trace()
            # c5 = self.adaAvgPool(c5) #lmj 1026 使不同尺寸的输入图片的输出相同->但这样会使小图片放大，对营养评估是否有影响未知
            # Top-down
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p3 = self._upsample_add(p4, self.latlayer2(c3))
            p2 = self._upsample_add(p3, self.latlayer3(c2))
            # Smooth
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            p2 = self.smooth3(p2)
            return p2, p3, p4, p5

    # 20211102
    def construct(self, x, bbox=None):
        # 20211102
        if self.yolobox:
            return self._forward_impl_bbox(x, bbox)
        else:
            return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, rgbd, bbox, **kwargs):
    model = ResNet(block, layers, rgbd, bbox, **kwargs)
    return model


def resnet50(pretrained=False, progress=True, rgbd=False, bbox=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, rgbd, bbox,
                   **kwargs)


def resnet101(pretrained=False, progress=True, rgbd=False, bbox=False, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, rgbd, bbox,
                   **kwargs)
