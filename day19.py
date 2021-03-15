# 6-2, 训练模型的3种方法

# Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。
# 有3类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环。
# 下面以minist数据集的分类模型的训练为例，演示这3种训练模型的风格。
# 其中类形式训练循环我们会使用torchkera.Model和torchkeras.LightModel这两种方法。

# 〇，准备数据

import torch
from torch import nn

import torchvision
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/", train=True, download=False, transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/", train=False, download=False, transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))

# 查看部分样本
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for i in range(9):
    img, label = ds_train[i]
    img = torch.squeeze(img)
    ax = plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

