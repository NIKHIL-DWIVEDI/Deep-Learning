import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import config
import torch
import math
import statistics
from torchvision.models.resnet import resnet18
from torch.utils.data import DataLoader

def train_val_split(n_val,n):
    len_val=int(n_val*n)
    idx=np.random.permutation(n)
    val_idx=idx[:len_val]
    train_idx=idx[len_val:]
    return train_idx,val_idx

class CNN(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.Model =  nn.Sequential(
            nn.Conv2d(in_channels=input_size,out_channels=8,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            # nn.Dropout(0.2),
            # nn.MaxPool2d(kernel_size=2,stride=1),

            nn.Conv2d(in_channels=8,out_channels=32,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            # nn.MaxPool2d(kernel_size=2,stride=1),

            nn.Conv2d(in_channels=32,out_channels=128,kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            # nn.MaxPool2d(kernel_size=2,stride=1),

            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            # nn.MaxPool2d(kernel_size=2,stride=1)

            nn.Conv2d(in_channels=64,out_channels=8,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.2)
        )

        self.fc1= nn.Linear(in_features= 288,out_features=64)
        self.fc2=nn.Linear(in_features=64,out_features=10)
        self.relu=nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        out = self.Model(x)
        temp=nn.Flatten()
        t=temp(out)
        fc1=self.fc1(t)
        fc1=self.relu(fc1)
        fc2=self.fc2(fc1)
        output= self.softmax(fc2)
        return output

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def train(model, epochs, train_dl,val_dl, learning_rate ):
    optimiser= torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    training_loss=[]
    training_acc=[]
    for epoch in range(epochs):
        ll=[]
        acc=[]
        for x,y in train_dl:
            x=x.to(config.DEVICE)
            y=y.to(config.DEVICE)
            out = model(x)
            loss = loss_func(out,y)
            ll.append(loss.item())
            acc.append(accuracy(out,y).item())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        training_loss.append(statistics.mean(ll))
        training_acc.append(statistics.mean(acc))
        print("Epoch : {} , Training Loss: {:4f} , Training Accuracy : {:4f}".format(epoch+1,training_loss[-1],training_acc[-1]) )

    return training_loss,training_acc


if __name__=="__main__":
    ## transform the data
    transform_image = transforms.Compose([transforms.ToTensor() , transforms.Normalize(mean=[0.5,0.5,0.5] , std=[0.5,0.5,0.5])])

    ### load the dataset
    train_data=CIFAR10(root='../MLP/data/',train=True,transform=transform_image,download=False)
    test_data=CIFAR10(root='../MLP/data/',train=False,transform=transform_image,download=False)

    ### split the train data into train and val i.e. 90 and 10
    train_idx,val_idx = train_val_split(0.1,len(train_data))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler=SubsetRandomSampler(val_idx)

    ### create DataLoader
    train_dl = DataLoader(train_data,batch_size=16,sampler= train_sampler,num_workers=2, pin_memory=True)
    val_dl =DataLoader(train_data,batch_size=16,sampler=val_sampler,num_workers=2,pin_memory=True)

    model = CNN(config.Input_size)
    model= model.to(config.DEVICE)
    # for x,y in train_dl:
    #     out = model(x)
    #     print(out)
    #     break
    resnet_model=resnet18(pretrained=True)
    # print(resnet_model)
    ## train the model
    training_loss,training_acc=train(resnet_model,config.EPOCHS,train_dl,val_dl,config.LEARNING_RATE)
    plt.plot(range(len(training_loss)),training_loss)
    plt.savefig("plot_results.png")


    




