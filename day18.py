# 六、Pytorch的高阶API
# Pytorch没有官方的高阶API。一般通过nn.Module来构建模型并编写自定义训练循环。
# 为了更加方便地训练模型，作者编写了仿keras的Pytorch模型接口：torchkeras，作为Pytorch的高阶API。
# 本章我们主要详细介绍Pytorch的高阶API如下相关的内容。
# 构建模型的3中方法（继承nn.Module基类，使用nn.Sequential,辅助应用模型容器）
# 训练模型的3种方法（脚本风格额，函数风格，，torchkeras.Model类风格）
# 使用GPU训练模型（单GPU训练，多GPU训练）

# 6-1，构建模型的3种方法
# 可以使用以下3种方式构建模型：
# 1，继承nn.Module基类构建自定义模型。
# 2，使用nn.Sequential按层顺序构建模型。
# 3，继承nn.Module基类构建模型并辅助应用模型容器进行封装(nn.Sequential, nn.ModuleList, nn.ModuleDict).
# 其中第1种方式最为常见，第2种方式最简单，第3种方式最为灵活也较为复杂。
# 推荐使用第1种方式构建模型。

import torch
from torch import nn
from torchkeras import summary

# 一，继承nn.Module基类构建自定义模型
# 以下是继承nn.Module基类构建自定义模型的一个范例。模型中的用到的层一般在__init__函数中定义，然后在forward方法中定义模型的正向传播逻辑。
