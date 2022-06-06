import torch.nn as nn
class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.avg = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,64),
            nn.Sigmoid()
        )

    def forward(self,x):
        y = self.avg(x)
        #print(x.shape)
        y = self.linear(y).view(-1,64,1,1)
        #print(x.shape)
        return x*y