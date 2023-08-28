#这个文件包含了网络模型的构建、训练、测试以及可视化
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataload import load_train_data, load_test_data

n_epochs = 9 #训练总轮数，在这里可以调整训练总轮数
batch_size_train = 64  #训练批次大小
batch_size_test = 1000  #测试批次大小
learning_rate = 0.01  #学习率
momentum = 0.5  #动量
log_interval = 10 #打印日志频率
random_seed = 1  #随机种子
torch.manual_seed(random_seed)

#调用加载训练和测试数据的函数
train_loader=load_train_data(batch_size_train)
test_loader=load_test_data(batch_size_test)

#定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

# 训练方法
def train(epoch):
    network.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 清空梯度信息
        output = network(data)  # 前向传播，得到预测输出
        loss = F.nll_loss(output, target)  # 计算损失（负对数似然损失）
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新参数
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())  # 记录训练损失
            train_counter.append((batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
            # 保存模型参数和优化器状态
            torch.save(network.state_dict(), './save/model.pth')
            torch.save(optimizer.state_dict(), './save/optimizer.pth')

# 测试方法
def test():
    network.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)  # 前向传播，得到预测输出
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 计算测试集损失
            pred = output.data.max(1, keepdim=True)[1]  # 获取预测类别
            correct += pred.eq(target.data.view_as(pred)).sum()  # 统计正确预测数量
    test_loss /= len(test_loader.dataset)  # 计算平均测试集损失
    test_losses.append(test_loss)  # 记录测试损失
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))  # 打印测试损失和准确率


# #训练和测试
train(1)
test()  # 不加这个，后面画图就会报错：x and y must be the same size
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()


#训练和测试结果可视化
  #测试损失和训练损失可视化
fig = plt.figure()
plt.plot(train_counter, train_losses, color='pink')
plt.scatter(test_counter, test_losses, color='blue')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig('./image/loss3.png')
  #测试准确率可视化 
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    output = network(example_data)
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
plt.savefig('./image/test.png')
plt.show()


