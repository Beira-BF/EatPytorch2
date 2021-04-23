# 6-3, 使用GPU训练模型
# 深度学习的训练过程常常非常耗时，一个模型训练几个小时是家常便饭，训练几天也是常有的事情，有时候甚至要训练几十天。
# 训练过程的耗时主要来自于两个部分，一部分来自数据准备，另一部分来自参数迭代。
# 当数据准备过程还是模型训练时间的主要瓶颈时，我们可以使用更多进程来准备数据。
# 当数据迭代过程成为训练时间的主要瓶颈时，我们通常的方法是应用GPU来进行加速。
# Pytorch中使用GPU加速模型非常简单，只要将模型和数据移动到GPU上，核心代码只有以下几行。


"""
# 定义模型

import torch

device = torch.device("cuda:0" if torch.cuda.is_availabel() else "cpu")
model.to(device)


# 训练模型


features = features.to(device)  # 移动数据到cuda
labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels
"""

# 如果要使用多个GPU训练模型，也非常简单。只需要在将模型设置为数据并行风格模型。则模型移动到GPU上之后，会在每一个GPU上拷贝一个副本，
# 并把数据平分到各个GPU上进行训练。核心代码如下。


"""
# 定义模型

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) # 包装为并行风格模型

# 训练模型

features = features.to(device) # 移动数据到cuda
labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels

"""

# 以下是一些和GPU有关的基本操作汇总
# 在Colab笔记本中： 修改->笔记本设置->硬件加速器 中选择GPU
# 注：以下代码只能在Colab上才能正确执行。
# 可点击如下链接，直接在colab中运行范例代码。
# 《torch使用gpu训练模型》
# https://colab.research.google.com/drive/1FDmi44-U3TFRCt9MwGn4HIj2SaaWIjHu?usp=sharing

import torch
from torch import nn

# 1, 查看gpu信息
if_cuda = torch.cuda.is_available()
print("if_cuda=", if_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=", gpu_count)

# 2,将张量在gpu和cpu间移动
tensor = torch.rand((100, 100))
tensor_gpu = tensor.to("cuda:0") # 或者 tensor_gpu = tensor.cuda()
print(tensor_gpu.device)
print(tensor_gpu.is_cuda)

tensor_cpu = tensor_gpu.to("cpu") # 或者 tensor_cpu = tensor_gpu.cpu()
print(tensor_cpu.device)

# 3,将模型中的全部张量移动到gpu上
net = nn.Linear(2,1)
print(next(net.parameters()).is_cuda)
net.to("cuda:0") # 将模型中的全部参数张量依次到GPU上，注意，无需重新赋值为 net = net.to("cuda:0")
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)


# 4, 创建支持多个gpu数据并行的模型
linear = nn.Linear(2,1)
print(next(linear.parameters()).device)

model = nn.DataParallel(linear)
print(model.device_ids)
print(next(model.module.parameters()).device)

# 注意保存参数时要指定保存model.module的参数
torch.save(model.module.state_dict(), "./data/model_parameter.pkl")

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load("./data/model_parameter.pkl"))


# 5, 清空cuda缓存

# 该方法在cuda超内存时十分有用
torch.cuda.empty_cache()


# 一，矩阵乘法范例
# 下面分别使用CPU和GPU作一个矩阵乘法，并比较其计算效率。

import time
import torch
from torch import nn

# 使用cpu
a = torch.rand((10000, 200))
b = torch.rand((200, 10000))
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()

print(toc-tic)
print(a.device)
print(b.device)

# 使用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.rand((10000, 200), device=device)  # 可以指定在GPU上创建张量
b = torch.rand((200, 10000)) # 也可以在CPU上创建张量后移动到GPU上
b = b.to(device) # 或者 b=b.cuda() if torch.cuda.is_avaliable() else b
tic = time.time()
c = torch.matmul(a, b)
toc = time.time()
print(toc-tic)
print(a.device)
print(b.device)

# 二，线性回归范例
# 下面对比使用CPU和GPU训练一个线性回归模型的效率
# 1，使用CPU

# 准备数据
n = 1000000 # 样本数量

X = 10*torch.rand([n,2])-5.0 # torch.rand是均匀分布
w0 = torch.tensor([[2.0, -3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 +torch.normal(0.0, 2.0, size=[n,1]) # @表示矩阵乘法，增加正态扰动


# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))
    # 正向传播
    def forward(self, x):
        return x@self.w.t() + self.b

linear = LinearRegression()

# 训练模型
optimizer = torch.optim.Adam(linear.parameters(), lr=0.1)
loss_func = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X)
        loss = loss_func(Y_pred, Y)
        loss.backward()
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch, "loss":loss.item()})
    toc = time.time()
    print("time used:", toc-tic)

train(500)

# 2, 使用GPU

# 准备数据
n = 1000000 # 样本数量

X = 10*torch.rand([n,2])-5.0 # torch.rand是均匀分布
w0 = torch.tensor([[2.0, -3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal(0.0, 2.0, size=[n,1])  # @表示矩阵乘法，增加正态扰动

# 移动到GPU上
print("torch.cuda.is_available() = ", torch.cuda.is_available())
X = X.cuda()
Y = Y.cuda()
print("X.device:", X.device)
print("Y.device:", Y.device)

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))

    # 正向传播
    def forward(self, x):
        return x@self.w.t() + self.b

linear = LinearRegression()

# 移动模型到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linear.to(device)

# 查看模型是否已经移动到GPU上
print("if no cuda:", next(linear.parameters()).is_cuda)

# 训练模型
optimizer = torch.optim.Adam(linear.parameters(), lr=0.1)
loss_func = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X)
        loss = loss_func(Y_pred, Y)
        loss.backward()
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch, "loss":loss.item()})
    toc = time.time()
    print("time used:", toc-tic)

train(500)

# 三，torchkeras.Model使用单GPU范例
# 下面演示使用torchkeras.Model来应用GPU训练模型的方法。
# 其对应的CPU训练模型代码参见《6-2，训练模型的3种方法》
# 本例仅需要在它的基础上增加一行代码，在model.compile时指定device即可。

# 1，准备数据

import torch
from torch import nn

import torchvision
from torchvision import transforms

import torchkeras

transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/", train=True, download=True, transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/", train=False, download=True, transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)
print(len(ds_train))
print(len(ds_valid))

# 查看部分样本
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for i in range(9):
    img, label = ds_train[i]
    img = torch.squeeze(img)
    ax=plt.subplot(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


# 2, 定义模型

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

net = CnnModel()
model = torchkeras.Model(net)
model.summary(input_shape=(1,32,32))

# 3,训练模型
from sklearn.metrics import accuracy_score

def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy())
    # 注意此处要将数据先移动到cpu上，然后才能转换成numpy数组

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.compile(loss_func = nn.CrossEntropyLoss(),
              optimizer=torch.optim.Adam(model.parameters(), lr=0.2),
              metrics_dict={"accuracy":accuracy}, device=device) # 注意此处compile时制定了device

dfhistory = model.fit(3, dl_train=dl_train, dl_val=dl_valid, log_step_freq=100)

# 4, 评估模型

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title("Training and validation "+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory, "loss")

plot_metric(dfhistory, "accuracy")

model.evaluate(dl_valid)

# 5,使用模型
model.predict(dl_valid)[0:10]

# 6,保存模型

# save the model parameters
torch.save(model.state_dict(), "model_parameter.pkl")

model_clone = torchkeras.Model(CnnModel())
model.clone.load_state_dict(torch.load("model_parameter.pkl"))
model_clone.compile(loss_func=nn.CrossEntropyLoss(),
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.02),
                    metrics_dict={"accuracy":accuracy}, device=device) # 注意此处compile时指定了device

model_clone.evaluate(dl_valid)

# 四，torchkeras.Model使用多GPU范例
# 注：以下范例需要在有多个GPU的机器上跑。如果在单GPU的机器上跑，也能跑通，但是实际上使用的单个GPU。
# 1，准备数据

import torch
from torch import nn

import torchvision
from torchvision import transforms

import torchkeras

transform = transforms.Compose([transforms.ToTensor()])
ds_train = torchvision.datasets.MNIST(root="./data/minist/", train=True, download=True, transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/", train=False, download=True, transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))


# 2,定义模型

class CnnModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

net = nn.DataParallel(CnnModel())  # Attention this line!!!
model = torchkeras.Model(net)

model.summary(input_shape=(1,32,32))

# 3, 训练模型

from sklearn.metrics import accuracy_score

def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy())
    # 注意此处要将数据先移动到cpu上，然后才能转换成numpy数组

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.compile(loss_func=nn.CrossEntropyLoss(),
              optimizer=torch.optim.Adam(model.parameters(), lr=0.02),
              metrics_dict={"accuracy":accuracy}, device=device) # 注意此处compile时制定了device

dfhistory = model.fit(3, dl_train=dl_train, dl_val=dl_valid, log_step_freq=100)

# 4, 评估模型

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics)+1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title("Training and validation "+metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory, "loss")

plot_metric(dfhistory, "accuracy")

model.evaluate(dl_valid)


# 5, 使用模型

model.predict(dl_valid)[0:10]


# 6, 保存模型

# save the model parameters

torch.save(model.net.module.state_dict(), "model_parameter.pkl")

net_clone = CnnModel()
net_clone.load_state_dict(torch.load("model_parameter.pkl"))

model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func=nn.CrossEntropyLoss(),
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.02),
                    metrics_dict={"accuracy":accuracy}, device=device)
model_clone.evaluate(dl_valid)

# 五，torchkeras.LightModel使用GPU/TPU范例
# 使用torchkeras.LightModel可以非常容易地将训练模式从cpu切换到单个gpu，多个gpu乃至多个tpu。
# 1，准备数据

import torch
from torch import nn

import torchvision
from torchvision import transforms

import torchkeras

transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/", train=True, download=True, transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/", train=False, download=True, transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))

# 2, 定义模型

import torchkeras
import pytorch_lightning as pl

class CnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Model(torchkeras.LightModel):

    # loss, and optional metrics
    def shared_step(self, batch)->dict:
        x,y = batch
        prediction = self(x)
        loss = nn.CrossEntropyLoss()(prediction,y)
        preds = torch.argmax(nn.Softmax(dim=1)(prediction), dim=1).data
        acc = pl.metrics.functional.accuracy(preds,y)
        dic = {"loss":loss, "acc":acc}
        return dic

    # optimizer, and optional lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)
        return {"optimizer": optimizer, "lr_scheduler":lr_scheduler}


pl.seed_everything(1234)
net = CnnNet()
model = Model(net)

# 3, 训练模型
ckpt_cb = pl.callbacks.ModelCheckpoint(monitor='val_loss')

# set gpus=0 will use cpu,
# set gpus=1 will use 1 gpu
# set gpus=2 will use 2 gpus
# set gpus=-1 will use all gpus
# you can also set gpus=[0,1] to use the given gpus
# you can even set tpu_cores=2 to use two tpus

trainer = pl.Trainer(max_epochs=10, gpus=2, callbacks=[ckpt_cb])

trainer.fit(model. dl_train, dl_valid)

# 4, 评估模型
import pandas as pd

history = model.history
dfhistory = pd.DataFrame(history)


import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics)+1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory, "loss")

plot_metric(dfhistory, "acc")

results = trainer.test(model, test_dataloaders=dl_valid, verbose=False)
print(results[0])

# 5, 使用模型

def predict(model, dl):
    model.eval()
    preds = torch.cat([model.forward(t[0].to(model.device)) for t in dl])

    result = torch.argmax(nn.Softmax(dim=1)(preds), dim=1).data
    return(result.data)

result = predict(model, dl_valid)

# 6, 保存模型
print(ckpt_cb.best_model_score)
model.load_from_checkpoint(ckpt_cb.best_model_path)

best_net = model.net
torch.save(best_net.state_dict(), "./data/net.pt")

net_clone = CnnNet()
net_clone.load_state_dict(torch.load("./data/net.pt"))
model_clone = Model(net_clone)
trainer = pl.Trainer()
result = trainer.test(model_clone, test_dataloaders=dl_valid, verbose=False)

print(result)
