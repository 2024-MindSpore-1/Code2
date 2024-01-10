import numpy as np
from mindspore import nn, Tensor

# loss_avg = nn.L1Loss(reduction="mean")
# loss_sum = nn.L1Loss(reduction="sum")
# loss_none = nn.L1Loss(reduction="none")
#
# input = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
# target = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
#
# print("loss avg:", loss_avg(input, target))
# print("loss sum:", loss_sum(input, target))
# print("loss none:", loss_none(input, target))

from loss.demo import MAELoss, MAELossTwo

pred = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
target = Tensor(np.array([0.1, 0.2, 0.2]).astype(np.float32))

loss = MAELoss()
print(loss(pred, target))

loss = MAELossTwo(reduction="mean")
print(loss(pred, target))

loss = MAELossTwo(reduction="sum")
print(loss(pred, target))

loss = MAELossTwo(reduction="none")
print(loss(pred, target))
