# 1.导入必要包
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
# from torch.autograd import Variable
torch.manual_seed(11)  #设置随机种子
# 2.生产数据
#%%
num_data = 1000
num_epoch = 1000

noise = init.normal_(torch.FloatTensor(num_data, 1), std=0.2)
# x = init.uniform_(torch.Tensor(num_data, 1), -10, 10)
x = init.uniform_(torch.Tensor(num_data, 1), -10, 10)
print(x)

y = 2 * x + 3
y_noise = 2 * (x + noise) + 3
# print(x,y,y_noise)

# 3. 模型和优化
model = nn.Linear(1, 1)
# output = model(Variable(x))
output = model(x)

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4.训练
loss_arr = []
label = y_noise
for i in range(num_epoch):
    output = model(x)
    optimizer.zero_grad()

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(loss)
        loss_arr.append(loss.item())

param_list = list(model.parameters())
# print(param_list[0].data,param_list[1].data)
# print(param_list[0],param_list[1])
print(param_list)
#以上三种都可以读取模型参数