import mindspore.nn as nn
from mindspore.ops import operations as ops
import sys

class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.theta_fc1 = nn.Dense(2, 64)
        self.theta_fc2 = nn.Dense(64, 64)
        self.theta_fc3 = nn.Dense(64, 64)
        self.theta_fc4 = nn.Dense(64, 4)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.leaky_relu(self.theta_fc1(x))
        x = self.leaky_relu(self.theta_fc2(x))
        x = self.leaky_relu(self.theta_fc3(x))
        x = self.sigmoid(self.theta_fc4(x))
        return x

class Nu(nn.Cell):
    def __init__(self):
        super(Nu, self).__init__()
        self.Nu_fc1 = nn.Dense(6, 64)
        self.Nu_fc2 = nn.Dense(64, 64)
        self.Nu_fc3 = nn.Dense(64, 1)
        self.cat = ops.Concat(axis=1)
        self.leaky_relu = nn.LeakyReLU()

    def construct(self, x, z):
        #print("2x: ", x.shape)
        #print("2z: ", z.shape)
        x = self.cat((x, z))
        x = self.leaky_relu(self.Nu_fc1(x))
        x = self.leaky_relu(self.Nu_fc2(x))
        x = self.Nu_fc3(x)  # log nu
        return x

class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        self.G_fc1 = nn.Dense(6, 64)
        self.G_fc2 = nn.Dense(64, 64)
        self.G_fc3 = nn.Dense(64, 64)
        self.G_fc4 = nn.Dense(64, 2)
        self.cat = ops.Concat(axis=1)
        self.leaky_relu = nn.LeakyReLU()

    def construct(self, x, eps):
        #print("1x: ", x.shape)
        #print("1eps: ", eps.shape)
        #sys.exit(0)
        x = self.cat((x, eps))
        x = self.leaky_relu(self.G_fc1(x))
        x = self.leaky_relu(self.G_fc2(x))
        x = self.leaky_relu(self.G_fc3(x))
        x = self.G_fc4(x)
        return x

