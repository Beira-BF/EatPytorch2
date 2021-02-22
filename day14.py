# 五、Pytorch的中阶API
# 我们将主要介绍Pytorch中的如下中阶API
# 数据管道
# 模型层
# 损失函数
# TensorBoard可视化
# 如果把模型比作一个房子，那么中阶API就是【模型之墙】


# 5-1，Dataset和DataLoader
# Pytorch通常使用Dataset和DataLoader这两个工具类来构建数据管道。
# Dataset定义了数据集的内容，它相当于一个类似列表的数据结构，具有确定的长度，能够用索引获取数据集中的元素。
# 而DatasetLoader定义了按batch加载数据集的方法，它是一个实现了__iter__方法的可迭代对象，每次迭代输出一个batch的数据。
# DataLoader能够控制batch的大小，batch中元素的采样方法，以及将batch结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据。
# 在绝大部分情况下，用户只需实现Dataset的__len__方法和__getitem__方法，就可以轻松构建自己的数据集，并用默认数据管道进行加载。


# 一、Dataset和DataLoader描述
# 1，获取一个batch数据的步骤
# 让我们考虑一下从一个数据集中获取一个batch的数据需要哪些步骤。
# （假定数据集的特征和标签分别表示为张量x和y，数据集可以表示为(X,Y),假定batch的大小为m）
# 1，首先我们要去欸的那个数据集的长度n。
# 结果类似：n = 1000.
# 2，然后我们从0到n-1的范围中抽样出m个数（batch的大小）。
# 假定m=4, 拿到的结果是一个列表，类似：indices=[1,4,8,9]
# 3,接着我们从数据集中去取这m个数对应下标的元素。
# 拿到的结果是一个元组列表，类似：samples=[(X[1],Y[1]), (X[4],Y[4]), (X[8],Y[8]), (X[9],Y[9])]
# 4,最后我们将结果整理成两个张量作为输出。
# 拿到的结果是两个张量，类似batch=(features, labels),
# 其中features = torch.stack([X[1], X[4], X[8], X[9]])
# labels = torch.stack([Y[1], Y[4], Y[8], Y[9]])

# 2, Dataset和DataLoader的功能分工
# 上述第1个步骤确定数据集的长度是由Dataset的__len__方法实现的。
# 第2个步骤从0到n-1的范围中抽样出m个数的方法是由DataLoader的sampleer和batch_sampler参数指定的。
# sampler参数指定单个元素抽样方法，一般无需用户设置，程序默认在DataLoader的参数shuffle=True时采用随机抽样，shuffle=False时采用顺序抽样。
# batch_sampler参数将多个抽样的元素整理成一个列表，一般无需用户设置，默认方法在DataLoader的参数drop_last=True时会丢弃数据集最后一个长度不能
# 被batch大小整除的批次，在drop_last=False时保留最后一个批次。
# 第3个步骤的核心逻辑根据下标取数据集中的元素，是由Dataset的__getitem__方法实现的。
# 第4个步骤的逻辑由DataLoader的参数cooate_fn指定。一般情况下也无需用户设置。

