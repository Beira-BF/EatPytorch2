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

# 正态分布随机
mean, std = 2,5
c = std*torch.randn((3,3))+mean
print(c)

# 整数随机排列
d = torch.randperm(20)
print(d)

# 特殊矩阵
I = torch.eye(3,3)  # 单位矩阵
print(I)
t = torch.diag(torch.tensor([1,2,3])) # 对角矩阵
print(t)

# 索引切片
# 张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。
# 可以通过索引和切片对部分元素进行修改。
# 此外，对于不规则的切片提取，可以使用torch.index_select, torch.masked_select, torch.take
# 如果要通过修改张量的某些元素得到新的张量，可是使用torch.where, torch.masked_fill, torch.index_fill

# 均匀随机分布
torch.manual_seed(0)
minval, maxval = 0, 10
t = torch.floor(minval + (maxval-minval)*torch.rand([5,5])).int()
print(t)

# 第0行
print(t[0])

# 倒数第一行
print(t[-1])

# 第1行第3列
print(t[1,3])
print(t[1][3])

# 第1行至第3行
print(t[1:4,:])

# 第1行至最后一行，第0列到最后一列每隔两列取一列
print(t[1:4,:4:2])

# 可以使用索引和切片修改部分元素
x = torch.tensor([[1,2],[3,4]], dtype=torch.float32, requires_grad=True)
x.data[1,:] =torch.tensor([0.0, 0.0])
print(x)

a = torch.arange(27).view(3,3,3)
print(a)

# 省略号可以表示多个冒号
print(a[...,1])

# 以上切片方式相对规则，对于不规则的切片提取，可以使用torch.index_select, torch.take, torch.gather, torch.masked_select.
# 考虑班级成绩册的例子，有4个班级，每个班级10个学生，每个学生7门科目成绩。可以用一个4x10x7的张量来表示。
minval = 0
maxval = 100
scores = torch.floor(minval + (maxval-minval)*torch.rand([4,10,7])).int()
print(scores)

# 抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
torch.index_select(scores, dim=1, index=torch.tensor([0,5,9]))

# 抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
q = torch.index_select(torch.index_select(scores, dim=1, index=torch.tensor([0,5,9])),
                       dim=2, index=torch.tensor([1,3,6]))
print(q)

# 抽取第0个班级第0个学生的第0门课程，第2个班级的第4个学生的第1门课程，第3个班级的第9个学生第6门课程成绩
# take将输入堪称一维数组，输出和index同形状
s = torch.take(scores, torch.tensor([0*10*7+0, 2*10*7+4*7+1, 3*10*7+9*7+6]))
print(s)

# 抽取分数大于等于80的分数（布尔索引）
# 结果是1维张量
g = torch.masked_select(scores, scores>=80)
print(g)

# 以上这些方法仅能提取张量的部分元素值，但不能更改张量的部分元素值得到新的张量。
# 如果要通过修改张量的部分元素值得到新的张量，可以使用torch.where, torch.index_fill和torch.masked_fill
# torch.where 可以理解为if的张量版本
# torch.index_fill的选取元素逻辑和torch.index_select相同。
# torch.masked_fill的选取元素逻辑和torch.masked_select相同。

# 如果分数大于60分，赋值成1，否则赋值成0
ifpass = torch.where(scores>60, torch.tensor(1), torch.tensor(0))
print(ifpass)

# 将每个班级第0个学生，第5个学生，第9个学生的全部成绩赋值成满分
torch.index_fill(scores, dim=1, index=torch.tensor([0,5,9]), value=100)
# 等价于scores.index_fill(dim=1, index=torch.tensor([0,5,9]),value=100)

# 将分数小于60分的分数赋值成60分
b = torch.masked_fill(scores, scores<60, 60)
# 等价于b=scores.masked_fill(scores<60, 60)
print(b)

# 三、维度变换
# 维度变换相关函数主要有torch.reshape(或者调用张量的view方法), torch.squeeze, torch.unsqueeze, torch.transpose
# torch.reshape 可以改变张量的形状。
# torch.squeeze可以减少维度。
# torch.unsqueeze可以增加维度。
# torch.transpose可以交换维度。

# 张量的view方法有时候会调用失败，可以使用reshape方法。

torch.manual_seed(0)
minval, maxval = 0, 255
a = (minval + (maxval-minval)*torch.rand([1,3,3,2])).int()
print(a.shape)
print(a)

# 改成（3，6）形状的张量
b = a.view([3,6]) # torch.reshape(a, [3,6])
print(b.shape)
print(b)

# 改回成[1,3,3,2]形状的张量
c = torch.reshape(b, [1,3,3,2]) # b.view([1,3,3,2])
print(c)

# 如果张量在某个维度上只有一个元素，利用torch.squeeze可以消除这个维度
# torch.unsqueeze的作用和torch.squeeze的作用相反

a = torch.tensor([[1.0, 2.0]])
s = torch.squeeze(a)
print(a)
print(s)
print(a.shape)
print(s.shape)

# 在第0维插入一个长度为1的一个维度

d = torch.unsqueeze(s, axis=0)
print(s)
print(d)

print(s.shape)
print(d.shape)

# torch.transpose可以交换张量的维度，torch.transpose常用于图片存储格式的变换上。
# 如果是二维的矩阵，通常会调用矩阵的转置方法matrix.t(), 等价于torch.transpose(matrix, 0, 1).

minval = 0
maxval = 255
# Batch, Height, Width, Channel
data = torch.floor(minval + (maxval-minval)*torch.rand([100, 256, 256, 4])).int()
print(data.shape)

