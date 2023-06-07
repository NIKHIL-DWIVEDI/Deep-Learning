import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

train_dataset=CIFAR10(root='/data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=CIFAR10(root='/data',train=False,transform=transforms.ToTensor(),download=True)

## split the train dataset into train and validation dataset
def train_val_split(n,n_val):
    val_len=int(n_val*n)
    train_len=n-val_len
    idx=np.random.permutation(n)
    return idx[:train_len], idx[train_len:]

train_idx,val_idx=train_val_split(len(train_dataset),0.1)

### training sampler and dataloader
train_sampler=SubsetRandomSampler(train_idx)
train_dataloader=DataLoader(dataset=train_dataset,batch_size=100,sampler=train_sampler,num_workers=2)

### validation sampler and dataloader
val_sampler=SubsetRandomSampler(val_idx)
val_dataloader=DataLoader(dataset=train_dataset,batch_size=100,sampler=val_sampler,num_workers=2)

input_size=3*32*32
classes=10

### create the Model
class MLP(nn.Module):
    def __init__(self,input_size,classes):
        super().__init__()
        self.l1=nn.Linear(in_features=input_size,out_features=input_size*4,bias=True)
        self.l2=nn.Linear(in_features=input_size*4,out_features=input_size*8,bias=True)
        self.l3=nn.Linear(in_features=input_size*8,out_features=input_size*16,bias=True)
        self.l4=nn.Linear(in_features=input_size*16,out_features=input_size*8,bias=True)
        self.l5=nn.Linear(in_features=input_size*8, out_features=input_size*2,bias=True)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.2)

    def forward(self,x):
        x1=self.relu(self.l1(x))
        x2=self.relu(self.l2(x1))
        x3=self.relu(self.l3(x2))
        x4=self.relu(self.l4(x3))
        x5=self.relu(self.l5(x4))
        x6=nn.Softmax(x4)
        return x5

model=MLP(input_size,classes)
print(model)