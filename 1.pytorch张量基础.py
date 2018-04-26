#%%
import torch

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
a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
print(a)
a_gpu = a.cuda()
print(a_gpu)
a_cpu = a_gpu.cpu()
print(a_cpu)

#%%
# 1.6 张量的尺寸
a = torch.FloatTensor(10, 12, 3)
print(a.size())
print(a.shape)
# torch.Size([10, 12, 3])
# 以上具有相同的作用

#%%
# 2 索引,切片,Joining,reshape
# 2.1 索引
torch.manual_seed(1122)
a = torch.rand(4, 3)
out = torch.index_select(a, 0, torch.LongTensor([0, 1]))
print(a, out)
# a:
# 0.8403  0.1383  0.5636
# 0.1963  0.2446  0.8257
# 0.0597  0.5320  0.4424
# 0.8020  0.1211  0.9157
# [torch.FloatTensor of size 4x3]
# out:
# 0.8403  0.1383  0.5636
# 0.1963  0.2446  0.8257
# [torch.FloatTensor of size 2x3]

a1 = a[:, 0]  #所有的行,第一列
a2 = a[[0, 1], :]  #前两行,所有列,等同于out
a3 = a[0:2, 0:2]  #前两行,前两列
print(a1, a2, a3)

x = torch.Tensor([[1, 2, 3], [3, 4, 5]])
#  1  2  3
#  3  4  5
# [torch.FloatTensor of size 2x3]
mask = torch.ByteTensor([[0, 0, 1], [0, 1, 0]])
# 0  0  1
# 0  1  0
# [torch.ByteTensor of size 2x3]
out = torch.masked_select(x, mask)
# 3
# 4
# [torch.FloatTensor of size 2]
print(x, mask, out)

#%%
# 2.2 Joining
x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])  #2x3
y = torch.FloatTensor([[-1, -2, -3], [-4, -5, -6]])  #2x3
z1 = torch.cat([x, y], dim=0)  #在第一个维度上进行叠加,4x3
#  1  2  3
#  4  5  6
# -1 -2 -3
# -4 -5 -6
# [torch.FloatTensor of size 4x3]
z2 = torch.cat([x, y], dim=1)  #在第二个维度上进行叠加,2x6
# 1  2  3 -1 -2 -3
# 4  5  6 -4 -5 -6
# [torch.FloatTensor of size 2x6]

print(z1, z2)
x_stack = torch.stack([x, x, x], dim=0)
#在一个新的维度上进行堆叠,dim=0,原来是2x3，这里在0维度上添加了三个x维度,
#所以维度上变为3x2x3
print(x_stack)
# (0 ,.,.) =
#   1  2  3
#   4  5  6

# (1 ,.,.) =
#   1  2  3
#   4  5  6

# (2 ,.,.) =
#   1  2  3
#   4  5  6
# [torch.FloatTensor of size 3x2x3]

#%%
# 2.3切片
x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])  #2x3
y = torch.FloatTensor([[-1, -2, -3], [-4, -5, -6]])  #2x3
z1 = torch.cat([x, y], dim=0)  #在第一个维度上进行叠加,4x3
z2 = torch.cat([x, y], dim=1)  #在第二个维度上进行叠加,2x6
x_1, x_2 = torch.chunk(z1, 2, dim=0)  #将z1在第一个维度上分成两个部分
y_1, y_2, y_3 = torch.chunk(z1, 3, dim=1)  #将z1在第二个维度上分成三个部分
print(z1, x_1, x_2, y_1, y_2, y_3)
# 1  2  3
# 4  5  6
# -1 -2 -3
# -4 -5 -6
# [torch.FloatTensor of size 4x3]

#  1  2  3
#  4  5  6
# [torch.FloatTensor of size 2x3]

# -1 -2 -3
# -4 -5 -6
# [torch.FloatTensor of size 2x3]

#  1
#  4
# -1
# -4
# [torch.FloatTensor of size 4x1]

#  2
#  5
# -2
# -5
# [torch.FloatTensor of size 4x1]

#  3
#  6
# -3
# -6
# [torch.FloatTensor of size 4x1]

x1, x2 = torch.split(z1, 2, dim=0)
y1 = torch.split(z1, 2, dim=1)
print(x1, x2, y1)

#  1  2  3
#  4  5  6
# [torch.FloatTensor of size 2x3]

# -1 -2 -3
# -4 -5 -6
# [torch.FloatTensor of size 2x3]
#  (
#  1  2
#  4  5
# -1 -2
# -4 -5
# [torch.FloatTensor of size 4x2]
# ,
#  3
#  6
# -3
# -6
# [torch.FloatTensor of size 4x1]
# )

#%%
# 2.4 压缩
x1 = torch.FloatTensor(10, 1, 3, 1, 4)
x2 = torch.squeeze(x1)
print(x1.shape, x2.shape, sep='\n')
x3 = torch.unsqueeze(x2, dim=1)
print(x3.shape)
# torch.Size([10, 1, 3, 1, 4])
# torch.Size([10, 3, 4])
# torch.Size([10, 1, 3, 4])

#%%
# 2.5 reshape
x1 = torch.arange(12)
x2 = x1.view(3, -1)
x3 = x1.view(2, -1, 2)
print(x1, x2, x3.shape)
#   0
#   1
#   2
#   3
#   4
#   5
#   6
#   7
#   8
#   9
#  10
#  11
# [torch.FloatTensor of size 12]

#   0   1   2   3
#   4   5   6   7
#   8   9  10  11
# [torch.FloatTensor of size 3x4]
#  torch.Size([2, 3, 2])

#%%
# 3 初始化
import torch.nn.init as init
import numpy as np
torch.manual_seed(1122)
x1 = init.uniform_(torch.FloatTensor(3, 4), a=0, b=9)
#(a,b)之间的均匀分布
x2 = init.normal_(torch.FloatTensor(3, 4), mean=1, std=0.2)
#均值为mean,标准差为std的正态分布
x3 = init.constant_(torch.FloatTensor(3, 4), np.pi)
print(x1, x2, x3)
#  7.5625  1.2449  5.0721  1.7665
#  2.2013  7.4314  0.5377  4.7879
#  3.9812  7.2177  1.0897  8.2409
# [torch.FloatTensor of size 3x4]

#  1.1815  1.1046  0.8012  1.1516
#  0.9263  0.7831  1.0669  0.9992
#  0.9842  0.8979  0.6505  1.2469
# [torch.FloatTensor of size 3x4]

#  3.1416  3.1416  3.1416  3.1416
#  3.1416  3.1416  3.1416  3.1416
#  3.1416  3.1416  3.1416  3.1416
# [torch.FloatTensor of size 3x4]

#%%
# 4 数学操作
# 4.1算术运算
x1 = torch.FloatTensor([[1, 2, 3], [3, 5, 5]])
x2 = torch.FloatTensor([[1, 2, 3], [3, 2, 1]])
print(x1, x2, x1 + x2, x1 - x2)
x3 = x1 + 10  #broadcasting
print(x3)
#乘法
x4 = torch.mul(x1, x2)  #等同于x1*x2
x5 = x1 * x2
x6 = x1 * 10
print(x4, x5, x6)  #broadcasting
#除法
x7 = x1 / x2
print(x7)

#%%
# 4.2 其他数学运算
x1 = torch.FloatTensor([[1, 2, 3], [3, 5, 5]])
x2 = x1**2  #对每个元素求平方
x3 = torch.exp(x1)  #求指数
x4 = torch.log(x1)  #求对数
print(x1, x2, x3, x4)

#%%
# 4.3 矩阵运算
x1 = torch.Tensor(torch.arange(1, 13).view(3, 4))
x2 = torch.ones(4, 5)
x3 = torch.mm(x1, x2)
print(x1, x2, x3)

#批量矩阵运算
x1 = torch.FloatTensor(10, 3, 4)
x2 = torch.FloatTensor(10, 4, 6)
x3 = torch.bmm(x1, x2)  #前面的10为批量
print(x3.shape)  #torch.Size([10, 3, 6])

#点乘
x1 = torch.FloatTensor((3, 4))
x2 = torch.FloatTensor((3, 2))

x3 = torch.dot(x1, x2)
print(x3)

#转置
x1 = torch.Tensor(torch.arange(1, 13).view(3, 4))
x2 = x1.t()
print(x1, x2)

#维度调换
x1 = torch.FloatTensor(10, 3, 4)
print(x1.shape, x1.transpose(1, 2).shape)
# torch.Size([10, 3, 4]) torch.Size([10, 4, 3])
print(x1.transpose(0, 1).transpose(1, 2).shape)  #torch.Size([3, 4, 10])
