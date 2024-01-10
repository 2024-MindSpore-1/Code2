from mindspore import nn, ops, Tensor


class NonLocal2D(nn.Cell):
    def __init__(self, in_channels):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1, has_bias=True)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1, has_bias=True)
        self. phi = nn.Conv2d(self.in_channels, self.inter_channels, 1, has_bias=True)
        self.conv_out = nn.Conv2d(self.in_channels, self.inter_channels, 1, has_bias=True)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = ops.matmul(theta_x, phi_x)
        pairwise_weight = ops.softmax(pairwise_weight, axis=-1)
        return pairwise_weight

    def construct(self, x):
        n, _, h, w = x.shape

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.transpose(0, 2, 1)

        # theta_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.transpose(0, 2, 1)

        # phi_x: [N, C, HxW]
        phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, 'embedded_gaussian')
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = ops.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.transpose(0, 2, 1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)

        return output


class BFP(nn.Cell):
    def __init__(self, in_channels, num_levels, refine_level=1):
        super(BFP, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.refine_level = refine_level

        assert 0 < self.refine_level < num_levels

        self.refine = NonLocal2D(self.in_channels)

    def construct(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].shape[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = ops.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = ops.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].shape[2:]
            if i < self.refine_level:
                residual = ops.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = ops.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])

        return tuple(outs)
