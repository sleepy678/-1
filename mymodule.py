import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from tensorboardX import SummaryWriter
#from torch.utils.tensorboard\
    #import tensorboard
#print(tensorboard.__version__)
from attention import  attention
class Convmodule(nn.Module):
    def __init__(self):
        super(Convmodule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,(3,3)),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,(3,3)),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Flatten(),#43264
            nn.Linear(43264, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.conv1(x)
        #print(x.shape)#1*3*112*112
        x = self.conv2(x)
        #print(x.shape)#1*32*55*55
        x = self.linear(x)
        #print(x.shape)#1*64*26*26
        return x
class Attentionmodule(nn.Module):
    def __init__(self):
        super(Attentionmodule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,(3,3)),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,(3,3)),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU()
        )
        self.attention = attention()
        self.linear = nn.Sequential(
            nn.Flatten(),#43264
            nn.Linear(43264, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.conv1(x)
        #print(x.shape)#1*3*112*112
        x = self.conv2(x)
        #print(x.shape)#1*32*55*55
        x = self.attention(x)
        x = self.linear(x)
        #print(x.shape)#1*64*26*26
        return x
class Linearmodule(nn.Module):
    def __init__(self):
        super(Linearmodule, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),#37632
            nn.Linear(37632, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.linear(x)
        return x
class xianxing(nn.Module):
    def __init__(self):
        super(xianxing, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),#37632
            nn.Linear(37632, 2)
        )

    def forward(self,x):
        x = self.linear(x)
        return x
if __name__ == "__main__":
    writer = SummaryWriter('./logs')
    mynet = mymodule()

    images = torch.randn(1, 1, 28, 28)
    writer.add_graph(mynet, images)
    writer.close()
    '''
    x = torch.rand(1,3,112,112)
    print(x.shape)
    print(mynet(x))'''


