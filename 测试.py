import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mymodule import Attentionmodule
import PIL.Image

def imshow(img, title=None):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

torch.set_printoptions(precision=4,sci_mode=False)
if __name__ == '__main__':
    paths = [f"cat{i}.png" for i in range(1,4)]
    path2 = [f"dog{i}.png" for i in range(1,4)]
    paths.extend(path2)
    #print(image2.shape)
    #print(type(image2))
    for path in paths:
        image = PIL.Image.open(path)
        transform = transforms.Compose([transforms.Resize((112,112)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5,0.5)])
        image = transform(image)
        #imshow(image)
        image = torch.unsqueeze(image,0)
        #print(image)
        #print(image.shape)
        net = Attentionmodule()
        path2 = "Attentionmodule.pth"
        net.load_state_dict(torch.load(path2))
        a = net(image)
        #print(a.shape)
        if a[0,0] > a[0,1]:
            print(f"{path}识别为猫")
        else:
            print(f"{path}识别为狗")