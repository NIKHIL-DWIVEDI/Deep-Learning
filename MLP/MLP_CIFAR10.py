import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch
import torch.nn.functional as F
import statistics


## split the train dataset into train and validation dataset
def train_val_split(n,n_val):
    val_len=int(n_val*n)
    train_len=n-val_len
    idx=np.random.permutation(n)
    return idx[:train_len], idx[train_len:]



### create the Model
class MLP(nn.Module):
    def __init__(self,input_size,classes):
        super().__init__()
        self.l1=nn.Linear(in_features=input_size,out_features=input_size*4,bias=True)
        self.l2=nn.Linear(in_features=input_size*4,out_features=input_size*8,bias=True)
        self.l3=nn.Linear(in_features=input_size*8,out_features=input_size*16,bias=True)
        self.l4=nn.Linear(in_features=input_size*16,out_features=input_size*8,bias=True)
        self.l5=nn.Linear(in_features=input_size*8, out_features=input_size*2,bias=True)
        self.l6=nn.Linear(in_features=input_size*2,out_features=10,bias=True)
        self.relu=nn.LeakyReLU()
        self.dropout=nn.Dropout(p=0.2)

    def forward(self,x):
        x1=self.dropout(self.relu(self.l1(x)))
        # print(x1.shape)
        x2=self.dropout(self.relu(self.l2(x1)))
        x3=self.relu(self.l3(x2))
        x4=self.relu(self.l4(x3))
        x5=self.dropout(self.relu(self.l5(x4)))
        x6=self.dropout(self.relu(self.l6(x5)))
        x7=F.softmax(x6,dim=1)
        return x7



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def train(model,epochs,train_dataloader,val_dataloader,learning_rate,batch_size):
    ##optimiser
    optimiser=torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    ##loss 
    # loss_func=F.cross_entropy
    training_loss=[]
    training_acc=[]
    
    for epoch in range(epochs):
        l=[]
        acc=[]
        for x,y in train_dataloader:
          x=x.to(device)
          y=y.to(device)
        #   print(x.shape)
          x=x.reshape((x.shape[0],3*16*16))
          out=model(x)
          loss=F.cross_entropy(out,y)
          l.append(loss.item())
          loss.backward()
          optimiser.step()
          optimiser.zero_grad()
          ac=accuracy(out,y)
          acc.append(ac.item())
          
        
        training_loss.append(statistics.mean(l))
        training_acc.append(statistics.mean(acc))
        print("Epoch : {} , Training Loss : {:.4f} , Training Accuracy : {:4f}".format(epoch+1,training_loss[-1],training_acc[-1]))

    ## print training loss 
    # plt.plot(range(Epochs),training_loss)
    # plt.xlabel("Epochs")
    # plt.ylabel("Training Loss")
    # plt.savefig("plot.png")
    return training_loss,training_acc


def validation(model,val_dataloader,batch_size):
    validation_loss=[]
    validation_acc=[]
    l=[]
    acc=[]
    for x,y in val_dataloader:
        x=x.to(device)
        y=y.to(device)
        x=x.reshape((x.shape[0],3*16*16))
        out=model(x)
        loss=F.cross_entropy(out,y)
        l.append(loss.item())
        ac=accuracy(out,y)
        acc.append(ac.item())
        
    validation_loss.append(statistics.mean(l))
    validation_acc.append(statistics.mean(acc))
    print(" Validation Loss : {:.4f} , Validation Accuracy : {:4f}".format(validation_loss[-1],validation_acc[-1]))
    return validation_loss,validation_acc



if __name__=="__main__":
    Epochs=5
    BATCH_SIZE=24
    IMAGE_SIZE=16
    input_size=3*IMAGE_SIZE*IMAGE_SIZE
    classes=10
    data_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    transforms.Resize((16,16))
    ])

    train_dataset=CIFAR10(root='./data',train=True,transform=data_transform,download=False)
    test_dataset=CIFAR10(root='./data',train=False,transform=data_transform,download=False)

    train_idx,val_idx=train_val_split(len(train_dataset),0.1)

### training sampler and dataloader
    train_sampler=SubsetRandomSampler(train_idx)
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=16,sampler=train_sampler,num_workers=2)

    ### validation sampler and dataloader
    val_sampler=SubsetRandomSampler(val_idx)
    val_dataloader=DataLoader(dataset=train_dataset,batch_size=16,sampler=val_sampler,num_workers=2)
    
    device=torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model=MLP(input_size,classes)
# model

    model.to(device)
    training_loss,training_acc=train(model,Epochs,train_dataloader,val_dataloader,0.001,BATCH_SIZE)
    validation_loss,validation_acc=validation(model,val_dataloader,BATCH_SIZE)
    plt.plot(range(Epochs),training_loss,label="Training Loass")
    # plt.plot(range(Epochs),validation_loss,label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("plot.png")


    #### test on validation data

