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

from mymodule import Convmodule,Attentionmodule,Linearmodule,xianxing
from attention import attention
from visualize import visualize
#图像处理
torch.set_printoptions(sci_mode=False)
def dataprocess(path,size):
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])
    dataset = datasets.ImageFolder(path,transform)
    # print(dataset)
    # print(dataset.class_to_idx)
    loader = torch.utils.data.DataLoader(dataset,batch_size = size,shuffle=True)
    return loader
path = "./dataset/kaggle/find_cats_and_dogs/train"
path2 = "./dataset/kaggle/find_cats_and_dogs/test"
batch_size = 8
train_loader = dataprocess(path,batch_size)#train
test_loader  = dataprocess(path2,batch_size)#test
#print(len(train_loader))
model = xianxing().cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,weight_decay=0.001)
device = torch.device('cuda')
loss = F.cross_entropy
#print(train_loader[0])
Loss2 = []
def train(model,loader,optimzer,loss,device,epoch):
    model.train()
    for index,(data,target) in enumerate(loader):
        data,target = data.to(device),target.to(device)

        out = model(data)
        #print("out,target", out,target)

        optimzer.zero_grad()
        Loss = loss(out,target)
        Loss.backward()
        optimzer.step()

        if (index + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
                epoch, (index + 1) * len(data), len(train_loader.dataset),\
                       100. * (index + 1) / len(train_loader), Loss.item()))
            Loss2.append(Loss.item())
    return Loss,out,target


def test(model,loader,optimzer,loss,device,epoch):
    model.eval()
    with torch.no_grad():
        for index,(data,target) in enumerate(loader):
            data,target = data.to(device),target.to(device)

            out = model(data)
            #print("target", target)
            #print("out",out)
            Loss = loss(out,target)
            if (index + 1) % 10 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
                    epoch, (index + 1) * len(data), len(train_loader.dataset),\
                           100. * (index + 1) / len(train_loader), Loss.item()))
                Loss2.append(Loss.item())
    return Loss

def adjust_learning_rate(optimizer, epoch):
    modellrnew = 0.1 if epoch<100 else 0.01
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
if __name__ == '__main__':

    epochs = 20
    for epoch in range(epochs):
        adjust_learning_rate(optimizer,epoch)
        Loss_train,out,target = train(model,train_loader,optimizer,loss,device,epoch)
        #print("out,target",out,target)
        Loss_val = test(model,test_loader,optimizer,loss,device,epoch)

        visualize(torch.ones(1) * epoch, [Loss_train.item()],"train_loss")
        visualize(torch.ones(1) * epoch, [Loss_val.item()],"val_loss")
    path = "./xianxing.pth"
    torch.save(model.state_dict(),path)
