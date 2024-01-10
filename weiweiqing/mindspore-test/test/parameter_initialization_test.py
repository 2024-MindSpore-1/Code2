from mindspore import nn
from mindspore.common.initializer import Normal

net = nn.Conv2d(3, 64, 3, weight_init=Normal(mean=0.2, sigma=0.01))
# net = nn.Conv2d(3, 64, 3, weight_init="normal")
