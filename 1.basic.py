import torch
a = torch.ones((2, 3))
b = torch.Tensor((2, 3)).cuda()
print(a, b)
