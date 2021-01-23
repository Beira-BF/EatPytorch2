# 3-3, 高阶API示范
# Pytorch没有官方的高阶API，一般需要用户自己实现训练循环、验证循环和预测循环。
# 作者通过仿照tf.keras.Model的功能对Pytorch的nn.Module进行了封装，设计了torchkeras.Model类，
# 实现了fit, validate, predict, summary方法，相当于用户自定义高阶API。
# 并示范了用它实现线性回归模型。
# 此外，作者还通过借用pytorch_lightning的功能，封装了类Keras接口的另外一种实现，即torchkeras.LightModel类。
# 并示范了用它实现DNN二分类模型。

import os
import datetime
from torchkeras import Model, summary

# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

# 一、线性回归模型
# 此范例我们通过继承torchkeras.Module模型接口，实现线性回归模型。
# 1，准备数据

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 样本数量
n = 400

# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0 # torch.rand是均匀分布
w0 = torch.tensor([[2.0], [-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal(0.0, 2.0, size=[n,1])  # @表示矩阵乘法，增加正态扰动

# 数据可视化
# plt.figure(figsize=(12,5))
# ax1 = plt.subplot(121)
# ax1.scatter(X[:,0], Y[:,0], c="b", label="samples")
# ax1.legend()
# plt.xlabel("x1")
# plt.ylabel("y", rotation=0)
#
# ax2 = plt.subplot(122)
# ax2.scatter(X[:,1], Y[:,0], c="g", label="samples")
# ax2.legend()
# plt.xlabel("x2")
# plt.ylabel("y", rotation=0)
# plt.show()

# 构建输入数据管道
ds = TensorDataset(X,Y)
ds_train, ds_valid = torch.utils.data.random_split(ds, [int(400*0.7), 400-int(400*0.7)])
dl_train = DataLoader(ds_train, batch_size=10, shuffle=True, num_workers=2)
dl_valid = DataLoader(ds_valid, batch_size=10, num_workers=2)

# 2,定义模型

# 继承用户自定义模型
from torchkeras import Model
class LinearRegression(Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(2,1)

    def forward(self, x):
        return self.fc(x)

model = LinearRegression()

model.summary(input_shape=(2,))

# 3, 训练模型
### 使用fit方法进行训练

def mean_absolute_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred-y_true))

def mean_absolute_percent_error(y_pred, y_true):
    absolute_percent_error = (torch.abs(y_pred-y_true)+1e-7)/(torch.abs(y_true)+1e-7)
    return torch.mean(absolute_percent_error)

model.compile(loss_func=nn.MSELoss(),
              optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
              metrics_dict={"mae":mean_absolute_error, "mape":mean_absolute_percent_error})

dfhistory = model.fit(200, dl_train=dl_train, dl_val=dl_valid, log_step_freq=20)

# 结果可视化
# w,b = model.state_dict()["fc.weight"], model.state_dict()["fc.bias"]
#
# plt.figure(figsize=(12,5))
# ax1 = plt.subplot(121)
# ax1.scatter(X[:,0], Y[:,0], c="b", label="samples")
# ax1.plot(X[:,0], w[0,0]*X[:,0]+b[0], "-r", linewidth=5.0, label="model")
# ax1.legend()
# plt.xlabel("x1")
# plt.ylabel("y", rotation=0)
#
# ax2 = plt.subplot(122)
# ax2.scatter(X[:,1], Y[:,0], c="g", label="samples")
# ax2.plot(X[:,1], w[0,1]*X[:,1]+b[0], "-r", linewidth=5.0, label="model")
# ax2.legend()
# plt.xlabel("x2")
# plt.ylabel("y", rotation=0)
#
# plt.show()

# 4, 评估模型
dfhistory.tail()

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, "bo--")
    plt.plot(epochs, val_metrics, "ro-")
    plt.title("Training and validation "+metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, "val_"+metric])
    plt.show()

# plot_metric(dfhistory,"loss")

# plot_metric(dfhistory,"mape")



# 5,使用模型
# 预测
dl = DataLoader(TensorDataset(X))
model.predict(dl)[0:10]

# 预测
model.predict(dl_valid)[0:10]

# 二，DNN二分类模型
# 此范例我们通过继承torchkeras.LightModel模型接口，实现DNN二分类模型。
# 1，准备数据

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchkeras
import pytorch_lightning as pl

# 正负样本数量
n_positive, n_negative = 2000, 2000

# 生成正样本，小圆环分布
r_p = 5.0 + torch.normal(0.0, 1.0, size=[n_positive, 1])
theta_p = 2*np.pi*torch.rand([n_positive, 1])
Xp = torch.cat([r_p*torch.cos(theta_p), r_p*torch.sin(theta_p)], axis=1)
Yp = torch.ones_like(r_p)

# 生成负样本，大圆环分布
r_n = 8.0 + torch.normal(0,1.0, size=[n_negative, 1])
theta_n = 2*np.pi*torch.rand([n_negative, 1])
Xn = torch.cat([r_n*torch.cos(theta_n), r_n*torch.sin(theta_n)], axis=1)
Yn = torch.zeros_like(r_n)

# 汇总样本
X = torch.cat([Xp, Xn], axis=0)
Y = torch.cat([Yp, Yn], axis=0)

# 可视化
plt.figure(figsize=(6,6))
plt.scatter(Xp[:,0], Xp[:,1], c="r")
plt.scatter(Xn[:,0], Xn[:,1], c="g")
plt.legend(["positive", "negative"])
plt.show()

ds = TensorDataset(X,Y)

ds_train, ds_valid = torch.utils.data.random_split(ds, [int(len(ds)*0.7), len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train, batch_size=100, shuffle=True, num_workers=2)
dl_valid = DataLoader(ds_valid, batch_size=100, num_workers=2)

# 2, 定义模型

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8)
        self.fc3 = nn.Linear(8,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y

class Model(torchkeras.LightModel):
    # loss, and optional metrics
    def shared_step(self,batch)->dict:
        x, y = batch
        prediction = self(x)
        loss = nn.BCELoss()(prediction, y)
        preds = torch.where(prediction>0.5, torch.ones_like(prediction), torch.zeros_like(prediction))
        acc = pl.metrics.functional.accuracy(preds,y)
        dic = {"loss":loss, "acc":acc}
        return dic
    # optimizer and optional lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)
        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler}

pl.seed_everything(1234)
net = Net()
model = Model(net)

torchkeras.summary(model, input_shape=(2,))

# 3, 训练模型

ckpt_cb = pl.callbacks.ModelICheckpoint(moniter='val_loss')

trainer = pl.Trainer(max_epochs=100, gpus=0, callbacks=[ckpt_cb])

trainer.fit(model, dl_train, dl_valid)

# 结果可视化
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
ax1.scatter(Xp[:,0], Xp[:,1], c="r")
ax1.scatter(Xn[:,0], Xn[:,1], c="g")
