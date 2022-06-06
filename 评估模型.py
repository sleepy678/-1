import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from mymodule import Convmodule,Attentionmodule,Linearmodule,xianxing
from main import dataprocess,test

torch.set_printoptions(sci_mode=False)

if __name__ == '__main__':
    batch_size = 8
    path = "./dataset/kaggle/find_cats_and_dogs/test"
    loader = dataprocess(path,batch_size)
    modulelist =["Convmodule","Attentionmodule","Linearmodule","xianxing"]
    modelname = modulelist[3]
    model = eval(f"{modelname}().cuda()")
    model.eval()
    model.load_state_dict(torch.load(f"{modelname}.pth"))
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1,weight_decay=0.001)
    device = torch.device('cuda')
    loss = F.cross_entropy

    data = F.softmax(torch.rand(8,2),dim=1)
    judge = torch.zeros(8)
    target = torch.ones(8)
    sum=0
    '''for j in range(8):
        if data[j,0] < data[j,1]:
            judge[j] = 1
        else:
            judge[j] = 0
        if judge[j] == target[j]:
            sum+=1
    print(data)
    print(judge)
    #print(target)
    print(sum)'''
    sum1 =0
    total = 2000
    #with torch.no_grad():
    for index,(data,target) in enumerate(loader):
        #print(data)
        out = model(data.cuda())
        target = target.tolist()
        #print("out",out)
        #print("target",target)
        judge = torch.zeros(8)
        sum=0
        for j in range(batch_size):
            if out[j, 0] < out[j, 1]:
                judge[j] = 1
            else:
                judge[j] = 0
            if judge[j] == target[j]:
                sum += 1
        sum1 +=sum
        #print("当前准确率为",sum/batch_size)
    print("准确率为",sum1/total)