import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor

# region create tensor

# from list
data = [i for i in range(5)]
x = Tensor(data)
# print(x)

# from numpy array
data = np.linspace(0, 10, 20)
y = Tensor(data)
# print(y)

# use tensor initializer
from mindspore.common.initializer import Normal, One

tensor1 = Tensor(shape=(2, 3), dtype=mindspore.int32, init=One())
tensor2 = Tensor(shape=(4, 5), dtype=mindspore.float32, init=Normal())
# print(tensor1)
# print(tensor2)

# inherit other tensor's attributes
x_ones = ops.ones_like(x)
y_zeros = ops.zeros_like(y)
# print(x_ones)
# print(y_zeros)

# endregion

# region tensor attributes

# print(f"shape={tensor2.shape}\n"
# 	  f"dtype={tensor2.dtype}\n"
# 	  f"itemsize={tensor2.itemsize}\n"
# 	  f"nbytes={tensor2.nbytes}\n"
# 	  f"ndim={tensor2.ndim}\n"
# 	  f"size={tensor2.size}\n"
# 	  f"strides={tensor2.strides}")

# endregion

# region tensor index

tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))

# print("First row: {}".format(tensor[0]))
# print("value of bottom right corner: {}".format(tensor[1, 1]))
# print("Last column: {}".format(tensor[:, -1]))
# print("First column: {}".format(tensor[..., 0]))

# endregion

# region tensor operations

x = Tensor(np.array([1, 2, 3]), mindspore.float32)
y = Tensor(np.array([4, 5, 6]), mindspore.float32)

output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

# print("add:", output_add)
# print("sub:", output_sub)
# print("mul:", output_mul)
# print("div:", output_div)
# print("mod:", output_mod)
# print("floordiv:", output_floordiv)

# Concat

data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)

# print(output)
# print("shape:\n", output.shape)

# Stack

data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.stack([data1, data2])

# print(output)
# print("shape:\n", output.shape)

# endregion

# region transform with numpy array

t = ops.ones(5, mindspore.float32)
# print(f"t: {t}")
n = t.asnumpy()
# print(f"n: {n}")

n = np.ones(5, dtype=np.float32)
# print(f"n: {n}")
t = Tensor.from_numpy(n)
# print(f"t: {t}")

# endregion
