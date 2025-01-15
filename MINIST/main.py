import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import * #全导入
from torch import nn
from torch.nn import Sequential,Module,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



train_data = torchvision.datasets.MNIST('./headdata',
                                          train = True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.MNIST('./headdata',
                                         train = False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练集的长度为{}'.format(train_data_size))
print('测试集的长度为{}'.format(test_data_size))

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

print(type(train_dataloader))

#拿几张图出来看看
# examples = enumerate(train_dataloader)
# batch_idx, (example_data, example_targets) = next(examples)
# fig = plt.figure()

# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title('Ground Truth: {}'.format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
# print(example_data.shape)

#调用网络
n = NN()

#损失函数 交叉熵损失
loss_fn = nn.CrossEntropyLoss()

#优化器 随机梯度下降
# 1e-2 =  1X10^(-2) = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(n.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epochs = 10
#损失
train_losses = []
train_counter = []
valid_losses = []
valid_counter = []

for epoch in range(1,epochs+1):
    for train_index, (inputs,labels) in enumerate(train_dataloader):
        outputs = n(inputs)
        optimizer.zero_grad()
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        if train_index % 10 == 0:
            train_losses.append(loss.item())
            counter_index = train_index *len(inputs) + (epoch-1) * len(train_dataloader.dataset)
            train_counter.append(counter_index)
            print('epoch: {}, [{}/{}({:.0f}%)], loss: {:.6f}'.format(
                epoch, train_index * len(inputs), len(train_dataloader.dataset),100 * (train_index*len(inputs)+(epoch-1)*len(train_dataloader.dataset))/(len(train_dataloader.dataset) * (epochs)), loss.item()))
        if train_index % 300 == 0:
            n.eval()
            valid_loss = []
            for valid_index,(inputs,labels) in enumerate(test_dataloader):
                outputs = n(inputs)
                loss = loss_fn(outputs,labels)
                valid_loss.append(loss.item())
            valid_losses.append(np.average(valid_loss))
            valid_counter.append(counter_index)
            print('validation loss: {:.6f} counter_index: {}'.format((np.average(valid_loss)), counter_index))
print('training ended')
#保存模型
torch.save(n,'trained.pth')


fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.plot(valid_counter, valid_losses, color='red')
plt.legend(['Train Loss', 'Valid Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Training images number')
plt.ylabel('Loss')
plt.show()





