# 1 导入必要包
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# 2 超参数
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
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # 原图片大小为3x28x28
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),  # 3x28x28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 3x14x14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),  # 3x14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 3x7x7
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# 6 实例化模型
cnn = ConvolutionalNeuralNetwork(num_classes)
cnn.to(device)

# 7 实例化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# 8 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = cnn(images)
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
    images = images.to(device)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    # correct +=(predicted==labels.to(device)).sum()
    correct += (predicted.cpu() == labels).sum()
print("卷积神经网络在10000个MNIST测试集图片的准确率为：%d %%" % (100 * correct / total))