#%%

import torch
from torch.autograd import Variable

# 1 张量及变量
x_tensor = torch.Tensor(torch.arange(1, 13).view(3, 4))
print(x_tensor)
#   1   2   3   4
#   5   6   7   8
#   9  10  11  12
# [torch.FloatTensor of size 3x4]
x_variable = Variable(x_tensor)
print(x_variable)
# Variable containing:
#   1   2   3   4
#   5   6   7   8
#   9  10  11  12
# [torch.FloatTensor of size 3x4]

# 封装张量
# x_variable.data->x_tensor
print(x_variable.data)
#   1   2   3   4
#   5   6   7   8
#   9  10  11  12
# [torch.FloatTensor of size

#%%
# 2 梯度
print(x_variable.grad)  #None
print(x_variable.requires_grad)  #False
x_variable = Variable(x_tensor, volatile=True)
# .volatile在推断过程中进行最小的内存使用，单个的volatile变量在整个图中,不需要梯度
print(x_variable.grad, x_variable.requires_grad, x_variable.volatile)
# None False True

#%%
# 3 图和变量
x = Variable(
    torch.FloatTensor(torch.arange(1, 13).view(3, 4)), requires_grad=True)
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
