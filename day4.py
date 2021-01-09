# 1-4, 时间序列数据建模流程范例
# 2020年发生的新冠肺炎疫情灾难给各国人民的生活造成了诸多方面的影响。
# 有的同学是收入上的，有的同学是感情上的，有的同学是心理上的，还有的同学是体重上的。
# 本文基于中国2020年3月之前的疫情数据，建立时间序列RNN模型，对中国的新冠肺炎疫情结束使劲按进行预测。

import os
import datetime
import importlib
import torchkeras

# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n", "=========="*8 + "%s"%nowtime)

# mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 一、准备数据
# 本文的数据集取自tushare，获取该数据集的方法参考了以下文章。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/covid-19.csv", sep="\t")
df.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"], figsize=(10,6))
plt.xticks(rotation=60)
plt.show()

dfdata = df.set_index("date")
dfdiff = dfdata.diff(periods=1).dropna()
dfdiff = dfdiff.reset_index("date")

# dfdiff.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"], figsize=(10,6))
# plt.xticks(rotation=60)
# dfdiff = dfdiff.drop("date", axis=1).astype("float32")
#
# dfdiff.head()

