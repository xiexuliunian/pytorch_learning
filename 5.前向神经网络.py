# 1 导入必要包
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# 2 超参数
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 3 准备数据集
train_dataset = dsets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    #ToTensor的作用是将 HxWxC变为 CxHxW，同时将灰度值范围由0~255变为0~1
    download=True)
test_dataset = dsets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

# 4 构造生成器
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 5 模型
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 6 实例化模型
net = FeedForwardNeuralNetwork(input_size, hidden_size, num_classes)
net.to(device)

# 7 实例化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# 8 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d] ,Step [%d/%d] ,Loss:%6.4f" %
                  (epoch + 1, num_epochs, i + 1,
                   len(train_dataset) // batch_size, loss.item()))
# 9 测试模型
correct = 0
total = 0
for images, labels in test_loader:
    images = images.reshape(-1, 28 * 28).to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    # correct +=(predicted==labels.to(device)).sum()
    correct += (predicted.cpu() == labels).sum()
print("前向传播神经网络在10000个MNIST测试集图片的准确率为：%d %%" % (100 * correct / total))
