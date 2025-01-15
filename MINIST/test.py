import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import * #全导入
from torch import nn
from torch.nn import Sequential,Module,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader
# 平均测试损失
test_loss_avg = 0
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# test数据加载
test_data = torchvision.datasets.MNIST('./headdata',
                                         train = False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_dataloader = DataLoader(test_data,batch_size = 64)
# 导入模型

my_model = torch.load('trained.pth')
my_model.eval()


def test(net, test_loader, loss_fn, device='cpu'):
    correct = 0
    total = 0
    test_loss = []
    with torch.no_grad():
        for train_idx, (inputs, labels) in enumerate(test_loader, 0):
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            test_loss.append(loss.item())
            index, value = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += int((value==labels).sum())
        test_loss_avg = np.average(test_loss)
        print('Total: {}, Correct: {}, Accuracy: {:.2f}%, AverageLoss: {:.6f}'.format(total, correct, (correct/total*100), test_loss_avg))

test(my_model,test_dataloader,loss_fn)
#输出 Total: 10000, Correct: 9746, Accuracy: 97.46%, AverageLoss: 0.071920