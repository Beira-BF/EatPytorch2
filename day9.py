# 3-2, 中阶API示范
# 下面的范例使用Pytorch的中阶API实现线性回归模型和DNN二分类模型
# Pytorch的中阶API主要包括各种模型层，损失函数，优化器，数据管道等等。

import os
import datetime

# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

# 一、线性回归模型
# 1，准备数据
import numpy
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 样本数量
n = 400

# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0  # torch.rand是均匀分布
w0 = torch.tensor([[2.0], [-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal(0.0, 2.0, size=[n,1]) # @表示矩阵乘法，增加正态扰动

# 数据可视化
plt.figure(figsize=(12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0], Y[:,0], c="b", label="samples")
plt.legend()
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1], Y[:,0], c="g", label="samples")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y", rotation=0)
plt.show()

# 构建输入数据管道
ds = TensorDataset(X,Y)
dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=2)

# 2,定义模型

model = nn.Linear(2,1) # 线性层
model.loss_func = nn.MSELoss()
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3,训练模型

def train_step(model, features, labels):
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad()
    return loss.item()

# 测试train_step效果
features, labels = next(iter(dl))
train_step(model, features, labels)

