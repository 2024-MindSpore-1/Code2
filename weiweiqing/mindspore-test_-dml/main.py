# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

#
# def print_hi(name):
#     # 在下面的代码行中使用断点来调试脚本。
#     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
#
#
# # 按间距中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助


# import numpy as np
# from mindspore import Tensor
# x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
# y = Tensor(np.array([1, 2]))
# z = y.unsqueeze(dim=0).t().expand_as(x)
# print(z)

# from mindspore import nn, Tensor
# import mindspore as ms
# import numpy as np
#
# net = nn.Dense(3, 5)
#
# x = Tensor(np.random.randn(10, 10, 3), ms.float32)
#
# y = net(x)
# print(y.shape)

# from mindspore import ops, dtype, nn, Tensor
# import mindspore as ms
# x = Tensor([[[[1, 2, 3], [4, 5, 6]]]], dtype=dtype.float32)
# print(x.shape)
# print(x)
#
# net = nn.LayerNorm([1,2,3])
# for param in net.get_parameters():
# 	print(param)
#
# print(net(x))

# from mindspore import nn
#
# net = nn.Conv2d(3, 64, 3, has_bias=True)
#
# for param in net.get_parameters():
# 	print(param.name)
# 	print(param.shape)


# from mindspore import ops, Tensor
#
# x = Tensor([1, 2, 3])
#
# y = ops.one_hot(x, 4, 0.8, .2)
#
# print(y.asnumpy())


# x = "goog"
# print(x.title())
import mindspore
import mindspore as ms
from mindspore import Tensor, ops
import numpy as np


# x = Tensor([1, 2, 3])
# y = x * 3
# print(y)

from loss.batch_miner import pdist

x = Tensor([[[1, 2], [3, 4]]], mindspore.float32)
y = x.clamp(min=1.5)
print(y)