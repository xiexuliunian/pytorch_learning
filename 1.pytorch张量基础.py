import torch
#%%
# 1 产生张量
# 1.1 产生随机数
torch.manual_seed(111)  #s设置random的种子
x = torch.rand(2, 3)  #均匀分布
print(x)
#0.7156  0.9140  0.2819
#0.2581  0.6311  0.6001

y = torch.randn(2, 3)  #正态分布
print(y)
#-1.7776  0.5832 -0.2682
#0.0241 -1.3542 -1.2677

#%%
# 1.2 zeros, ones, arange
a = torch.zeros(2, 3)  #后面跟具体size

b = torch.ones(4, 5)  #后面跟具体size

c = torch.zeros_like(b)  #后面跟相似的张量

d = torch.arange(0, 3, step=0.1)
#开始,结束,步长;包含头尾共31个数
e = torch.linspace(0, 3, 10)
#开始,结束,共多少个,包含头尾
print(a, b, c, d, e)

#%%
# 1.3 张量数据类型
a = torch.FloatTensor(2, 3)
#相当于初始化为近视于0的随机数
# size=(2,3)
b = torch.FloatTensor((2, 3.5))  #2.0000,3.5000
c = torch.FloatTensor([2, 3])
#b,c具有一样的作用
d = b.type_as(torch.IntTensor())  #2,3
# 将张量从FloatTensor转变为IntTensor
print(a, b, c, d)

#%%
# 1.4 numpy和张量的互相转换
import numpy as np
a = np.array((1, 2, 3.5))
b = torch.from_numpy(a)  #numpy转张量
print(a, b)
c = b.numpy()  #张量转numpy
print(c)

#%%
# 1.5 不同设备上的张量
a=torch.FloatTensor([[1,2,3],[4,5,6]])
print(a)
a_gpu=a.cuda()
print(a_gpu)
a_cpu=a_gpu.cpu()
print(a_cpu)