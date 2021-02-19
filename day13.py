# 4-3, nn.functional和nn.Module
import os
import datetime

# 打印时间
def prirntbar():
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n"+"=========="*8 + "%s"%nowtime)

# 一，nn.functional和nn.Module
# 前面我们介绍了Pytorch的张量的结构操作和数学运算中的一些常用API
# 利用这些张量的API我们可以构建出神经网络相关的组件（如激活函数，模型层，损失函数）
# Pytorch和神经网络相关的功能组件大多都封装在torch.nn模块下。
# 这些功能组件的绝大部分既有函数形式实现，也有类形式实现。
# 其中nn.functional（一般引入后改名为F）有各种功能组件的函数实现。例如：
#（激活函数）
# F.relu
# F.sigmoid
# F.tanh
# F.softmax
# (模型层)
# F.linear
# F.conv2d
# F.max_pool2d
# F.dropout2d
# F.embedding
# (损失函数)
# F.binary_cross_entropy
# F.mse_loss
# F.cross_entropy

# 为了便于对参数进行管理，一般通过继承nn.Module转换成为类的实现形式，并直接封装在nn模块下。例如：
# (激活函数）
# nn.ReLU
# nn.Sigmoid
# nn.Tanh
# nn.Softmax
# (模型层)
# nn.Linear
# nn.Conv2d
# nn.MaxPool2d
# nn.Dropout2d
# nn.Embedding
# (损失函数)
# nn.BCELoss
# nn.MSELoss
# nn.CrossEntropyLoss
# 实际上nn.Module除了可以管理其引用的各种参数，还可以管理其引用的子模块，功能十分强大。

# 二、使用nn.Module来管理参数
# 在Pytorch中，模型的参数是需要被优化器训练的，因此，通常要设置参数为require_grad = True的张量。
# 同时，在一个模型中，往往有许多的参数，要手动管理这些参数并不是一件容易的事情。
# Pytorch一般将参数用nn.Parameter来表示，并且用nn.Module来管理其结构下的所有参数。

import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# nn.Parameter具有requires_grad = True 属性
w = nn.Parameter(torch.randn(2,2))
print(w)
print(w.requires_grad)

# nn.ParameterList可以将多个nn.Parameter组成一个列表
params_list = nn.ParameterList([nn.Parameter(torch.rand(8,i)) for i in range(1,3)])
print(params_list)
print(params_list[0].requires_grad)

params_dict = nn.ParameterDict({"a": nn.Parameter(torch.rand(2,2)),
                                "b": nn.Parameter(torch.zeros(2))})
print(params_dict)
print(params_dict["a"].requires_grad)

# 可以用Module将它们管理起来
# module.parameters()返回一个生成器，包括其结构下的所有parameters

module = nn.Module()
module.w = w
module.params_list = params_list
module.params_dict = params_dict

num_param = 0
for param in module.parameters():
    print(param, "\n")
    num_param = num_param + 1
print("number of Parameters = ", num_param)

# 实践当中，一般通过继承nn.Module来构建模块类，并将所有含有需要学习的参数的部分放在构造函数中。

# 以下范例为Pytorch中nn.Linear的源码的简化版本
# 可以看到它将需要学习的参数放在了__init__构造函数中，并在forward中调用F.linear函数来实现计算逻辑。

class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

