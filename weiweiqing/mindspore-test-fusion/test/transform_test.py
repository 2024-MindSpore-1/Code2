from download import download
from mindspore.dataset import vision, transforms

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from download import download


# region  vision module with PIL image input or numpy array
#
# url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/banana.jpg"
# download(url, './banana.jpg', replace=False)
#
# img_ori = Image.open("banana.jpg").convert("RGB")
# print(f"Image Type: {type(img_ori)}, Image Shape: { img_ori.size}")
#
# img = vision.Resize(size=300)(img_ori)
# print(f"Image Type: {type(img)}, Image Shape: { img.size}")
#
# img = vision.CenterCrop(size=(280, 280))(img)
# print(f"Image Type: {type(img)}, Image Shape: { img.size}")
#
# img = vision.Pad(40)(img)
# print(f"Image Type: {type(img)}, Image Shape: { img.size}")
#
# plt.subplot(1, 2, 1)
# plt.title("original image")
# plt.imshow(img_ori)
# plt.subplot(1, 2, 2)
# plt.title("transformed image")
# plt.imshow(img)
# plt.show()

# endregion


# region transform module with numpy array input

data = np.array([1, 2, 3, 4, 5])
# print(transforms.Fill(0)(data))

label_array = np.array([2, 3])
print(transforms.OneHot(num_classes=5)(label_array))

# endregion
