# 1.导入必要包
#%%
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 2.超参数
input_size = 784  #28x28=784，将每个像素点作为一个神经元输入对待
num_classes = 10  #MNIST数据集类别为10类
num_epoch = 10
batch_size = 100
learning_rate = 0.001

# 3.准备MNIST数据集
train_dataset = dsets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    #ToTensor的作用是将 HxWxC变为 CxHxW，同时将灰度值范围由0~255变为0~1
    download=True)
test_dataset = dsets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

# 4.数据迭代器
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 5.模型构建
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LogisticRegression(input_size, num_classes)
model.to(device)

# 6.准则和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 7.训练
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # 前向+后向+优化
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' %
                  (epoch + 1, num_epoch, i + 1,
                   len(train_dataset)// batch_size, loss.item()))
        
correct = 0
total = 0
for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    
print('模型在10000张测试集的准确率为: %d %%' % (100 * correct / total))
# epoch为10时，模型在10000张测试集的准确率为: 85 %