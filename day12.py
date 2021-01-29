# 4-2, 张量的数学运算
# 张量的操作主要包括张量的结构操作和张量的数学运算
# 张量的结构操作诸如：张量创建，索引切片，维度变换，合并分割
# 张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制
# 本篇我们介绍张量的数学运算。
# 本篇文章部分内容参考如下博客：https://blog.csdn.net/duan_zhihua/article/details/82526505

# 一、标量运算
# 张量的数学运算符可以分为标量运算符、向量运算符、以及矩阵运算符。
# 加减乘除乘方，以及三角函数，指数，对数等常见函数，逻辑比较运算符等都是标量运算符。
# 标量运算符的特点是对张量实施逐元素运算。
# 有些标量运算符对常用的数学运算符进行了重载，并且支持类似numpy的广播特性。

import torch
import numpy as np

a = torch.tensor([[1.0, 2], [-3, 4.0]])
b = torch.tensor([[5.0, 6], [7.0, 8.0]])
print(a+b) # 运算符重载

print(a-b)
print(a*b)
print(a/b)
print(a**2)
print(a**0.5)

print(a%3) # 求模

print(a//3) # 地板除法

print(a>=2) # torch.ge(a,2) #ge:greater_equal缩写

print((a>=2)&(a<=3))

print((a>=2)|(a<=3))

print(a==5) # torch.eq(a,5)

a = torch.tensor([1.0, 8.0])
b = torch.tensor([5.0, 6.0])
c = torch.tensor([6.0, 7.0])

d = a + b + c
print(d)

print(torch.max(a,b))

print(torch.min(a,b))

x = torch.tensor([2.6, -2.7])

print(torch.round(x))  # 保留整数部分，四舍五入
print(torch.floor(x))  # 保留整数部分，向下归整
print(torch.ceil(x))   # 保留整数部分，向上归整
print(torch.trunc(x))  # 保留整数部分，向0归整

x = torch.tensor([2.6, -2.7])
print(torch.fmod(x,2)) # 作除法取余数
print(torch.remainder(x,2))  # 作除法取剩余的部分，结果恒正

# 幅值裁剪
x = torch.tensor([0.9, -0.8, 100.0, -20.0, 0.7])
y = torch.clamp(x, min=-1, max=1)
z = torch.clamp(x, max=1)

print(y)
print(z)

