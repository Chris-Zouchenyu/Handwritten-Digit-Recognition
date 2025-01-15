import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#from main import *
# test数据加载
test_data = torchvision.datasets.MNIST('./headdata',
                                         train = False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_dataloader = DataLoader(test_data,batch_size = 64)
# 模型导入
my_model = torch.load('trained.pth')
my_model.eval()
# 测试开始
examples = enumerate(test_dataloader)
batch_idx, (inputs, targets) = next(examples)
with torch.no_grad():
    outputs = my_model(inputs)
    fig = plt.figure()
    for i in range(0,6):
        # 在前一百个测试样本当中找寻预测结果与标签不一致的样本
        # if(targets[i].item() != outputs.data.max(1, keepdim=True)[1][i].item()):
        #     print(i)
        plt.subplot(3, 2, i+1)
        plt.imshow(inputs[i][0], cmap='gray', interpolation='none')
        plt.title('Truth: {} Prediction: {}'.format(targets[i], outputs.argmax(dim = 1)[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()