# 5-4,TensorBoard可视化
# 在我们的炼丹过程中，如果能够使用丰富的图像来展示模型的结构，指标的变化，参数的分布，输入的形态等信息，无疑会提升我们对问题的洞察力，
# 并增加许多炼丹的乐趣。

# TensorBoard正是这样一个神奇的炼丹可视化辅助工具。它原是TensorFlow的小弟，但它也能够很好地和Pytorch进行配合。甚至在Pytorch中使用
# TensorBoard比TensorFlow中使用TensorBoard还要来的更加简单和自然。

# Pytorch中利用TensorBoard可视化的大概过程如下：
# 首先在Pytorch中指定一个目录创建一个torch.utils.tensorboard.SummaryWriter日志写入器。
# 然后根据需要可视化的信息，利用日志写入器将相应信息日志写入我们指定的目录。
# 最后就可以传入日志目录作为参数启动TensorBoard，然后就可以在TensorBoard中愉快地看片了。

# 我们主要介绍Pytorch中利用TensorBoard进行如下方面信息的可视化的方法。
# 可视化模型结构：writer.add_graph
# 可视化指标变化：writer.add_scalar
# 可视化参数分布：writer.add_histogram
# 可视化原始图像：writer.add_image或writer.add_images
# 可视化人工绘图：writer.add_figure

# 一、可视化模型结构

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchkeras import Model, summary

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y

net = Net()
print(net)

summary(net, input_shape=(3,32,32))

writer = SummaryWriter('./data/tensorboard')
writer.add_graph(net, input_to_model=torch.rand(1,3,32,32))
writer.close()

from tensorboard import notebook
notebook.list()

notebook.start("--logdir ./data/tensorboard")

# 二，可视化指标变化
# 有时候在训练过程中，如果能够实时动态地查看loss和各种metric地变化曲线，那么无疑可以帮助我们更加直观地了解模型地训练情况。
# 注意，writer.add_scalar仅能对标量地值地变化进行可视化。因此它一般用于对loss和metric地变化进行可视化分析。

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# f(x) = a*x**2 + b*x + c 的最小值
x = torch.tensor(0.0, requires_grad=True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x], lr=0.01)

def f(x):
    result = a*torch.pow(x,2) + b*x + c
    return (result)

writer = SummaryWriter('./data/tensorboard')
for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
    writer.add_scalar("x", x.item(), i) # 日志中记录x在第step i的值
    writer.add_scalar("y", y.item(), i) # 日志中记录y在第step i的值

writer.close()

print("y=", f(x).data, ";", "x=", x.data)

# 三，可视化参数分布
# 如果需要对模型的参数（一般非标量）在训练过程中的变化进行可视化，可以使用writer.add_histogram.
# 它能够观测张量值分布的直方图随训练步骤的变化趋势。

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# 创建正态分布的张量模拟参数矩阵
def norm(mean, std):
    t = std*torch.randn((100,20))+mean
    return t

writer = SummaryWriter('./data/tensorboard')
for step, mean in enumerate(range(-10,10,1)):
    w = norm(mean, 1)
    writer.add_histogram("w", w, step)
    writer.flush()
writer.close()

# 可视化原始图像
# 如果我们作图像相关的任务，也可以将原始的图片在tensorboard中进行可视化展示。
# 如果只写入一张图片信息，可以使用writer.add_image.
# 如果要写入多张图片信息，可以使用writer.add_images.
# 也可以使用torchvision.utils.make_grid将多张图片拼成一张图片，然后用writer.add_image写入。
# 注意，传入的是代表图片信息的Pytorch中的张量数据。

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transformers, datasets

