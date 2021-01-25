# 四、Pytorch的低阶API
# Pytorch的低阶API主要包括张量操作，动态计算图和自动微分
# 如果把模型比作一个房子，那么低阶API就是【模型之砖】
# 在低阶API层次上，可以把Pytorch当作一个增强版的numpy来使用。
# Pytorch提供的方法比numpy更全面，运算速度更快，如果需要的话，还可以使用GPU进行加速。
# 前面几章我们对低阶API已经有了一个整体的认识，本章我们将重点详细介绍张量操作和动态计算图。
# 张量的操作主要包括张量的结构操作和张量的数学运算。
# 张量结构操作诸如：张量创建，索引切片，纬度变换，合并分割。
# 张量的数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。
# 动态就算图我们将主要介绍动态计算图的特性，计算图中的Function，计算图与反向传播。

# 4-1， 张量的结构操作
# 张量的操作主要包括张量的结构操作和张量的数学运算。
# 张量结构操作诸如：张量创建，索引切片，纬度变换，合并分割
# 张量数学运算主要有：标量运算，向量运算，矩阵运算，另外我们会介绍张量运算的广播机制。
# 本篇我们介绍张量的结构操作哦。

# 一、创建张量
# 张量创建的许多方法和numpy中创建array的方法很像。

import numpy as np
import torch

a = torch.tensor([1,2,3], dtype=torch.float)
print(a)

b = torch.arange(1, 10, step=2)
print(b)

# torch.linspace 为线性等分向量，开始、结束、分割点数
c = torch.linspace(0.0, 2*3.14, 10)
print(c)

d = torch.zeros((3,3))
print(d)

a = torch.ones((3,3), dtype=torch.int)
b = torch.zeros_like(a, dtype=torch.float)
print(a)
print(b)

torch.fill_(b,5)
print(b)

# 均匀随机分布
torch.manual_seed(0)
minval, maxval = 0, 10
a = minval + (maxval-minval)*torch.rand([5])
print(a)

# torch.rand()生成0-1之间的随机数
print(torch.rand([5]))

# 正态分布随机
b = torch.normal(mean=torch.zeros(3,3), std=torch.ones(3,3))
print(b)

