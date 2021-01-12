# 二、Pytorch的核心概念
# Pytorch是一个基于Python的机器学习库。它广泛应用于计算机视觉，自然语言处理等深度学习领域。是目前和TensorFlow分庭抗礼的深度学习
# 框架，在学术圈颇受欢迎。
# 它主要提供了以下两种核心功能：
# 1， 支持GPU加速的张量计算。
# 2， 方便优化模型的自动微分机制。
# Pytorch的主要优点：
# 简洁易懂： Pytorch的API设计的相当简洁一致。基本上就是tensor，autograd，nn三级封装。学习起来非常容易。有一个这样的段子，说
# Tensorflow的设计哲学是 Make it complicated, Keras的设计哲学是 Make it complicated and hide it, 而Pytorch的设计哲学是
# Keep it simple and stupid.
# 便于调试： Pytorch采用动态图，可以像普通Python代码一样进行调试。不同于Tensorflow, Pytorch的报错说明通常很容易看懂。有一个这样的
# 段子，说你永远不可能从TensorFlow的报错说明中找到它出错的原因。
# 强大高效： Pytorch提供了非常丰富的模型组件，可以快速实现想法。并且运行速度很快。目前大部分深度学习相关的Paper都是用Pytorch
# 实现的，有些研究人员表示，从使用TensorFlow转换为使用Pytorch之后，他们的睡眠好多了，头发比以前浓密了，皮肤也比以前光滑了。

# 俗话说，万丈高楼平地起，Pytorch这座大厦也有它的地基。
# Pytorch底层最核心的概念是张量，动态计算图以及自动微分。


# 2-1， 张量数据结构
# Pytorch的基本数据结构是张量Tensor。张量即多维数组。Pytorch的张量和numpy中的array很类似。
# 本节我们主要介绍张量的数据类型、张量的维度、张量的尺寸、张量和numpy数组等基本概念。

# 一、张量的数据类型
# 张量的数据类型和numpy.array基本一一对应，但是不支持str类型。
# 包括：
"""
torch.float64(torch.double),
torch.float32(torch.float),
torch.float16,
torch.int64(torch.long),
torch.int32(torch.int),
torch.int16,
torch.int8,
torch.uint8,
torch.bool
"""
# 一般神经网络建模使用的都是torch.float32类型。

import numpy as np
import torch

# 自动推断数据类型

i = torch.tensor(1);print(i, i.dtype)
x = torch.tensor(2.0);print(x, x.dtype)
b = torch.tensor(True);print(b, b.dtype)

# 指定数据类型

i = torch.tensor(1, dtype=torch.int32);print(i, i.dtype)
x = torch.tensor(2.0, dtype=torch.double);print(x, x.dtype)

# 使用特定类型构造函数

i = torch.IntTensor(1);print(i, i.dtype)
x = torch.Tensor(np.array(2.0));print(x, x.dtype) # 等价于torch.FloatTensor
y = torch.FloatTensor(np.array(2.0));print(y, y.dtype)
b = torch.BoolTensor(np.array([1, 0, 2, 0]));print(b, b.dtype)

# 不同类型进行转换
i = torch.tensor(1);print(i, i.dtype)
x = i.float();print(x,x.dtype) # 调用float方法转换成浮点类型
y = i.type(torch.float);print(y, y.dtype) # 使用type函数转换成浮点类型
z = i.type_as(x);print(z,z.dtype) # 使用type_as方法转换成某个Tensor相同类型

# 二、张量的维度
# 不同类型的数据可以用不同维度(dimension)的张量来表示。
# 标量为0维张量，向量为1维张量，矩阵为2维张量。
# 彩色图像有rgb三个通道，可以表示为3维张量。
# 视频还有时间维，可以表示为4维张量。
# 可以简单地总结为：有几层中括号，就是多少维的张量。

scalar = torch.tensor(True)
print(scalar)
print(scalar.dim())  # 标量，0维张量


vector = torch.tensor([1.0, 2.0, 3.0, 4.0])  # 向量，1维张量
print(vector)
print(vector.dim())

matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 矩阵，2维张量
print(matrix)
print(matrix.dim())

tensor3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],[[5.0, 6.0], [7.0, 8.0]]]) # 3维张量
print(tensor3)
print(tensor3.dim())

tensor4 = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
                        [[[5.0, 5.0], [6.0, 6.0]], [[7.0, 7.0], [8.0, 8.0]]]])  # 4维张量
print(tensor4)
print(tensor4.dim())

# 三、张量的尺寸
# 可以使用shape属性或者size()方法查看张量在每一维的长度。
# 可以使用view方法改变张量的尺寸。
# 如果view方法改变尺寸失败，可以使用reshape方法。

scalar = torch.tensor(True)
print(scalar, scalar.dtype)
print(scalar.size())
print(scalar.shape)

vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(vector,vector.dtype)
print(vector.size())
print(vector.shape)