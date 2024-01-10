import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):

    def __init__(self, num_class=4):
        super(Decoder, self).__init__()
        self.theta_fc1 = nn.Linear(2, 64)
        self.theta_fc2 = nn.Linear(64, 64)
        self.theta_fc3 = nn.Linear(64, 64)
        self.theta_fc4 = nn.Linear(64, num_class)

    def forward(self, x):
        x = F.leaky_relu(self.theta_fc1(x))
        x = F.leaky_relu(self.theta_fc2(x))
        x = F.leaky_relu(self.theta_fc3(x))
        x = torch.sigmoid(self.theta_fc4(x))
        return x


class Nu(nn.Module):

    def __init__(self, num_class=4):
        super(Nu, self).__init__()
        self.Nu_fc1 = nn.Linear(num_class+2, 64)
        self.Nu_fc2 = nn.Linear(64, 64)
        self.Nu_fc3 = nn.Linear(64, 1)

    def forward(self, x, z):
        x = torch.cat((x, z), dim=1)

        x = F.leaky_relu(self.Nu_fc1(x))
        x = F.leaky_relu(self.Nu_fc2(x))
        x = self.Nu_fc3(x)  # log nu
        return x


class Encoder(nn.Module):

    def __init__(self, num_class=4):
        super(Encoder, self).__init__()
        self.G_fc1 = nn.Linear(num_class+2, 64)
        self.G_fc2 = nn.Linear(64, 64)
        self.G_fc3 = nn.Linear(64, 64)
        self.G_fc4 = nn.Linear(64, 2)
        #self.dropout = nn.Dropout(0.1) #tzy

    def forward(self, x, eps):
        #x = self.dropout(x) #tzy 
        x = torch.cat((x, eps), dim=1)

        x = F.leaky_relu(self.G_fc1(x))
        x = F.leaky_relu(self.G_fc2(x))
        x = F.leaky_relu(self.G_fc3(x))
        x = self.G_fc4(x)
        return x
