import torchvision
import torch
from torch import nn
from torch.nn import Sequential, Module, Conv2d, MaxPool2d, Flatten, Linear

# 搭建神经网络，十分类的网络
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2),  # 添加padding以保持输出尺寸
            nn.ReLU(),  # 添加激活函数
            MaxPool2d(kernel_size=2, stride=2),  # 通常池化层的stride与kernel_size相同
            Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2),  # 添加padding
            nn.ReLU(),  # 添加激活函数
            MaxPool2d(kernel_size=2, stride=2),  # 通常池化层的stride与kernel_size相同
            Flatten(),  # 展平输出以匹配全连接层的输入
            Linear(8 * 7 * 7, 64),  # 注意这里的输入特征数应该是展平后的尺寸（假设输入是28x28，经过两次2x2的池化后变为7x7）
            nn.ReLU(),  # 添加激活函数
            Linear(64, 10),  # 输出层，10个类别
        )

    def forward(self, x):
        x = self.model1(x)
        return x

# 测试模型写的对不对

# n = NN()
# input = torch.ones((64, 1, 28, 28))  # 注意输入应该是四维的，包括批次大小和通道数
# output = n(input)
# print(output.shape)  # 应该输出 torch.Size([64, 10])