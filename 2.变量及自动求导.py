#%%

import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1 张量及变量
x_tensor = torch.tensor(np.arange(1,13).reshape(3,4),dtype=torch.float32,requires_grad=True)
# x_tensor=x_tensor.to(device)
y_tensor =torch.tensor(torch.arange(1,13).reshape(3,4))#现在可以使用reshape了
# x_tensor.requires_grad_(requires_grad=True)
print(x_tensor,"\n",y_tensor)
#   1   2   3   4
#   5   6   7   8
#   9  10  11  12
# [torch.FloatTensor of size 3x4]


#%%
# 2 梯度
print(x_tensor.grad)  #None
print(x_tensor.requires_grad)  #False


#%%
# 3 图和变量
x = x_tensor
print(x)
y = x**2 + 4 * x
z = 2 * y + 3
print(x.requires_grad,y.requires_grad,z.requires_grad)
# True True True
z.sum().backward()#方向传播的值应该是个标量
print(x.grad)#仍然是个变量
# Variable containing:
#  12  16  20  24
#  28  32  36  40
#  44  48  52  56
# [torch.FloatTensor of size 3x4]

print(x.grad==4*x+8)
# Variable containing:
#  1  1  1  1
#  1  1  1  1
#  1  1  1  1
# [torch.ByteTensor of size 3x4]
