import numpy as np
import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore import Parameter, Tensor, ParameterTuple

x = Parameter(default_input=Tensor(np.arange(2*3).reshape(2, 3), ms.float32), name="x")
y = Parameter(default_input=initializer(init="ones", shape=[1, 2, 3], dtype=ms.float32), name="y")
z = Parameter(default_input=2.0, name="z")

params = ParameterTuple((x, y, z))

params_copy = params.clone(prefix="params_copy")

print(params)
print(params_copy)
