import torch

ten = torch.ones(size=(1, 64, 64, 250))
print(ten.shape)

x = ten.squeeze(-1)
print(x.shape)

x = x.reshape(-1, 4096)
print(x.shape)
