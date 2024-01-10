from .others import *
from .bfp import BFP


class Resnet101_concat(nn.Cell):
    def __init__(self):
        super(Resnet101_concat, self).__init__()
        self.refine = BFP(512, 4)

        self.smooth1 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.smooth2 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.smooth3 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.smooth4 = nn.Conv2d(512, 512, kernel_size=3, stride=1)

        self.ca0 = ChannelAttention(512)
        self.sa0 = SpatialAttention()
        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(512)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(512)
        self.sa3 = SpatialAttention()

        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_3 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_4 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Dense(2048, 1024)

        self.calorie = nn.SequentialCell(nn.Dense(1024, 1024), nn.Dense(1024, 1))
        self.mass = nn.SequentialCell(nn.Dense(1024, 1024), nn.Dense(1024, 1))
        self.fat = nn.SequentialCell(nn.Dense(1024, 1024), nn.Dense(1024, 1))
        self.carb = nn.SequentialCell(nn.Dense(1024, 1024), nn.Dense(1024, 1))
        self.protein = nn.SequentialCell(nn.Dense(1024, 1024), nn.Dense(1024, 1))
        # self.LayerNorm = nn.LayerNorm(2048)

    # 4向量融合，一个result
    def construct(self, rgb, rgbd):
        # pdb.set_trace()
        cat0 = ops.concat((rgb[0], rgbd[0]), 1)  # torch.Size([16, 512, 64, 64])
        cat1 = ops.concat((rgb[1], rgbd[1]), 1)  # torch.Size([16, 512, 32, 32])
        cat2 = ops.concat((rgb[2], rgbd[2]), 1)  # torch.Size([16, 512, 16, 16])
        cat3 = ops.concat((rgb[3], rgbd[3]), 1)  # torch.Size([16, 512, 8, 8])
        # BFP
        cat0, cat1, cat2, cat3 = self.refine(tuple((cat0, cat1, cat2, cat3)))
        # 两种模态的特征融合后再一起过个卷积
        cat0 = self.smooth1(cat0)  # torch.Size([16, 512, 64, 64])
        cat1 = self.smooth1(cat1)  # torch.Size([16, 512, 32, 32])
        cat2 = self.smooth1(cat2)  # torch.Size([16, 512, 16, 16])
        cat3 = self.smooth1(cat3)  # torch.Size([16, 512, 8, 8])
        # CMBA
        cat0 = self.ca0(cat0) * cat0
        cat0 = self.sa0(cat0) * cat0
        cat1 = self.ca1(cat1) * cat1
        cat1 = self.sa1(cat1) * cat1
        cat2 = self.ca2(cat2) * cat2
        cat2 = self.sa2(cat2) * cat2
        cat3 = self.ca3(cat3) * cat3
        cat3 = self.sa3(cat3) * cat3

        cat0 = self.avgpool_1(cat0)
        cat1 = self.avgpool_2(cat1)
        cat2 = self.avgpool_3(cat2)
        cat3 = self.avgpool_4(cat3)

        # pdb.set_trace()

        cat_input = ops.stack([cat0, cat1, cat2, cat3], axis=1)  # torch.Size([16, 4, 512, 1, 1])
        input = cat_input.view(cat_input.shape[0], -1)  # torch.Size([N, 5, 1024]) N =16(bz) 11(最后batch图片不足)
        # 20210907 #验证能否加速收敛
        # pdb.set_trace()
        # input = self.LayerNorm(input)
        #
        input = self.fc(input)
        input = ops.relu(input)  # torch.Size([16, 2048]) 添加原因：faster rcnn 也加了

        results = []
        results.append(self.calorie(input).squeeze())
        results.append(self.mass(input).squeeze())
        results.append(self.fat(input).squeeze())
        results.append(self.carb(input).squeeze())
        results.append(self.protein(input).squeeze())

        return results
