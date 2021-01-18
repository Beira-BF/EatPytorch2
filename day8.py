# 三、Pytorch的层次结构
# 本章我们介绍Pytorch中5个不同的层次结构：即硬件层，内核层，低阶API，中阶API，高阶API【torchkeras】。并以线性回归
# 和DNN二分类模型为例，直观对比展示在不同层级实现模型的特点。
# Pytorch的层次结构从低到高可以分成如下五层。
# 最底层为硬件层，Pytorch支持CPU，GPU加入计算资源池。
# 第二层为C++实现的内核。
# 第三层为Python实现的操作符，提供了封装C++内核的低级API指令，主要包括各种张量操作算子、自动微分、变量管理，
# 如torch.tensor, torch.cat, torch.autograd.grad, nn.Module. 如果把模型比作一个房子，那么第三层API就是【模型之砖】。
# 第四层为Python实现的模型组件，对低级API进行了函数封装，主要包括各种模型层，损失函数，优化器，数据管道等等。
# 如torch.nn.Linear, torch.nn.BCE, torch.optim.Adam, torch.utils.data.DataLoader.如果把模型比作一个房子，那么第四层
# API就是【模型之墙】。
# 第五层为Python实现的模型接口。Pytorch没有官方的高阶API。为了便于训练模型，作者仿照keras中的模型接口，使用了
# 不到300行代码，封装了Pytorch的高阶模型接口torchkears.Model。如果把模型比作一个房子，那么第五层API就是模型本身，
# 即【模型之屋】。

# 3-1，低阶API示范
# 下面的范例使用Pytorch的低阶API实现线性回归模型和DNN二分类模型。
# 低阶API主要包括张量操作，计算图和自动微分。

import os
import datetime

# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+ "=========="*8 + "%s"%nowtime)

# 一、线性回归模型
# 1，准备数据
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn

# 样本数量
n = 400

# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0  # torch.rand是均匀分布
w0 = torch.tensor([[2.0], [-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal(0.0, 2.0, size=[n,1])  # @表示矩阵乘法，增加正态扰动

# 数据可视化

plt.figure(figsize=(12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0].numpy(), Y[:,0].numpy(), c="b", label="samples")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1].numpy(), Y[:,0].numpy(), c="g", label="samples")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y", rotation=0)
plt.show()

# 构建数据管道迭代器
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = torch.LongTensor(indices[i:min(i+ batch_size, num_examples)])
        yield features.index_select(0, indexs), labels.index_select(0, indexs)

# 测试数据管道效果
batch_size = 8
(features, labels) = next(data_iter(X,Y,batch_size))
print(features)
print(labels)

