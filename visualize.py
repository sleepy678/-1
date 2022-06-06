import torch
from visdom import Visdom
import numpy as np

# 新建名为'demo'的环境
def visualize(t,arr,name):
    viz = Visdom(env=name)
    viz.line(X=t,Y=arr,win = 'train_loss',update = 'append',)
    # 窗口的名称
    opts = dict(title=name)  # 图像的标例

