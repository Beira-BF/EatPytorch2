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
