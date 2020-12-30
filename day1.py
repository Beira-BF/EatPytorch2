# 一、Pytorch的建模流程
# 使用Pytorch实现神经网络的一般流程包括：
# 1，准备数据
# 2，定义模型
# 3，训练模型
# 4，评估模型
# 5，使用模型
# 6，保存模型

# 对新手来说，其中最苦难的部分实际上是准备数据过程。
# 我们在实践中通常会遇到的数据类型包括结构化数据，图片数据，文本数据，时间序列数据。
# 我们将分别以titanic生存预测问题，cifar2图片分类问题，imdb电影评论分类问题，国内新冠疫情结束时间预测问题为例，演示应用Pytorch对这
# 四类数据的建模方法。


# 1-1, 结构化数据建模流程范例
import os
import datetime

# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=============="*8 + "%s"%nowtime)

# mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 一、准备数据
# titanic数据集的目标是根据乘客信息预测他们在Titanic号撞击冰山沉没后能否生存。
# 结构化数据一般会使用Pandas中的DateFrame进行预处理。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

dftrain_raw = pd.read_csv('./data/titanic/train.csv')
dftest_raw = pd.read_csv('./data/titanic/test.csv')
dftrain_raw.head(10)

# 字段说明：
# Survived:0代表死亡，1代表存活【y标签】
# Pclass：乘客所持票类，有三种值（1，2，3）【转换成onehot编码】
# Name:乘客姓名【舍去】
# Sex：乘客性别【转换成bool特征】
# Age：乘客年龄（有缺失）【数值特征，添加"年龄是否缺失"作为辅助特征】
# SibSp：乘客兄弟姐妹/配偶的个数（整数值）【数值特征】
# Parch：乘客父母/孩子的个数（整数值）【数值特征】
# Ticket：票号（字符串）【舍去】
# Fare：乘客所持票的价格（浮点数，0-500不等）【数值特征】
# Cabin：乘客所在船舱（有缺失）【添加"所在船舱是否缺失"作为辅助特征】
# Embarked：乘客登船港口：S、C、Q（有缺失）【转换成onehot编码，四维度S，C，Q，nan】

# 利用Pandas的数据可视化功能我们可以简单地进行探索性数据分析EDA（Exploratory Data Analysis）。

### 在作图的时候，只能显示一张图片，所以需要把其他的图片注释掉

# label分布情况
# ax = dftrain_raw['Survived'].value_counts().plot(kind='bar',
#                                                  figsize=(12,8), fontsize=15, rot=0)
# ax.set_ylabel('Counts', fontsize=15)
# ax.set_xlabel('Survived', fontsize=15)
# plt.show()

# 年龄分布情况
# ax = dftrain_raw['Age'].plot(kind='hist', bins=20, color='purple',
#                              figsize=(12,8), fontsize=15)
# ax.set_ylabel('Frequency', fontsize=15)
# ax.set_xlabel('Age', fontsize=15)
# plt.show()

# 年龄和label的相关性
# ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density',
#                                                     figsize=(12,8), fontsize=15)
# ax.legend(['Survived==0', 'Survived==1'], fontsize=12)
# ax.set_ylabel('Density', fontsize=15)
# ax.set_xlabel('Age', fontsize=15)
# plt.show()

# 下面为正式的数据预处理
def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    # Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_'+ str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp, Parch, Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return (dfresult)

x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw[['Survived']].values

x_test = preprocessing(dftest_raw).values
y_test = dftest_raw[['Survived']].values

print("x_train.shape=", x_train.shape)
print("x_test.shape=", x_test.shape)

print("y_train.shape=", y_train.shape)
print("y_test.shape=", y_test.shape)

# 进一步使用DataLoader和TensorDataset封装成可以迭代的数据管道。
dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float()),
                      shuffle=True, batch_size=8)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float()),
                      shuffle=False, batch_size=8)

# 测试数据管道
for features, labels in dl_train:
    print(features, labels)
    break

