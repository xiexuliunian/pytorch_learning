# 1 导入必要包
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import datetime

# 2 超参数
epoch_number = 80
num_classes = 10
batch_size = 100
learning_rate = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3 图像预处理
transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

# 4 CIFAR-10数据集
train_dataset = dsets.CIFAR10(
    root="./data/", train=True, transform=transform, download=True)

test_dataset = dsets.CIFAR10(
    root='./data/', train=False, transform=transforms.ToTensor())

# 5 构造生成器
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 6 网络模型


# 6.1 3x3的卷积
def conv3x3(int_channels, out_channels, stride=1):
    return nn.Conv2d(
        int_channels, out_channels, kernel_size=3, stride=stride, padding=1)


# 6.2 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 6.3 残差网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# 7 实例化网络、损失函数、优化器

resnet = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)
resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)

# 8 训练

for epoch in range(epoch_number):
    prev_time = datetime.datetime.now()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #前向后向优化
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        #用于显示训练过程
        if (i + 1) % 100 == 0:
            epoch_str=("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f lr: %.5f" %
                  (epoch + 1, 80, i + 1, len(train_dataset) // batch_size,
                   loss.item(),learning_rate))
            print(epoch_str,time_str,sep=" ")
        
        # prev_time = cur_time

    #设置学习率衰减
    if (epoch + 1) % 20 == 0:
        learning_rate /= 5.0
        optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)

# 9 测试
correct=0
total=0
for images,labels in test_loader:
    images =images.to(device)
    outputs=resnet(images)
    _,predicted=torch.max(outputs.data,1)
    total +=labels.size(0)
    correct +=(predicted.cpu()==labels).sum()

print("深度残差网络在CIFAR-10数据集上的测试准确率为：%d %%" %(100*correct/total))
