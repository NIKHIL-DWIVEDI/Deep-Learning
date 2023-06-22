import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb
import config
import torch.nn as nn
import torchvision.datasets as dataset
from torchvision.transforms import transforms
import PIL
import torchvision.utils
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter 
from torchvision.utils import save_image
#### function to break train data into train and val


def train_val_split(n,n_val):
    val_len=int(n*n_val)
    train_len = n-val_len
    idx = np.random.permutation(n)
    return idx[:train_len],idx[train_len:]

#### create the model i.e. discriminator and generator #####
### generator model
class Generator(nn.Module):
    def __init__(self,inputDim=100, outChannel=1):
        super().__init__()
        self.l1 = nn.ConvTranspose2d(in_channels=inputDim,out_channels=128,kernel_size=3,stride=2)
        self.l2 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=3)
        self.l3 =nn.ConvTranspose2d(in_channels=64,out_channels=16,kernel_size=2,stride=3)
        self.l4 =nn.ConvTranspose2d(in_channels=16,out_channels=outChannel,kernel_size=3,stride=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.BN1=nn.BatchNorm2d(128)
        self.BN2=nn.BatchNorm2d(64)
        self.BN3=nn.BatchNorm2d(16)
        
    def forward(self,x):
        x=self.l1(x)
        x=self.relu(x)
        x=self.BN1(x)
        x=self.l2(x)
        x=self.relu(x)
        x=self.BN2(x)
        x=self.l3(x)
        x=self.relu(x)
        x=self.BN3(x)
        x=self.l4(x)
        x=self.tanh(x)
        return x

### discriminator model
class Discriminator(nn.Module):
    def __init__(self,depth,alphs=0.2):
        super().__init__()
        self.l1=nn.Conv2d(in_channels=depth,out_channels=32,kernel_size=3,stride=2)
        self.l2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2)
        self.leakyrelu= nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=2304,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=1)

    def forward(self,x):
        x=self.l1(x)
        x=self.leakyrelu(x)
        x=self.l2(x)
        x=self.leakyrelu(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.leakyrelu(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        return x
    

# custom weights initialization called on ``netG`` and ``netD``
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

### train function
def train(train_dl,gen_model,dis_model,batch_size,epochs,gen_lr,dis_lr):
    gen_loss=[]
    dis_loss=[]
    gen_acc=[]
    dis_acc=[]
    gen_optimiser=torch.optim.Adam(gen_model.parameters(),lr=gen_lr,betas=(0.5,0.999))
    dis_optimiser=torch.optim.Adam(dis_model.parameters(),lr=dis_lr,betas=(0.5,0.999))
    
    loss_func = nn.BCELoss()
    # loss_func_1 = BCEWithLogitsLoss()
    step=0

    for epoch in range(epochs):
        g_loss=[]
        d_loss=[]
    
        for image, label in train_dl:
            image=  image.to(config.DEVICE)
            label=label.to(config.DEVICE)
            
            ## train discriminator
            noise=torch.randn(image.shape[0],100,1,1).to(config.DEVICE)
            real_targets=torch.ones(image.shape[0],1).to(config.DEVICE)
            fake_targets=torch.zeros(image.shape[0],1).to(config.DEVICE)
            fake_image = gen_model(noise)  #G(z)
            dis_out = dis_model(image)      #D(x)
            # print(fake_image,dis_out)
            # print(dis_out.shape, real_targets.shape)
            loss1= loss_func(dis_out,real_targets) # max log D(x)
            disc_fake = dis_model(fake_image) #D(G(z))
            loss2= loss_func(disc_fake,fake_targets) # max 1-log D(G(z))
            loss= (loss1+loss2)/2
            dis_model.zero_grad()
            d_loss.append(loss.item())
            loss.backward(retain_graph=True)
            dis_optimiser.step()
            # dis_optimiser.zero_grad()

            ## train discriminator
            gen_out = dis_model(fake_image)
            loss3=loss_func(gen_out,real_targets)
            gen_model.zero_grad()
            loss3.backward()
            g_loss.append(loss3.item())
            gen_optimiser.step()
            # gen_optimiser.zero_grad()


        print(f"Epoch [{epoch}/{epochs}] \Loss D: {d_loss[-1]:.4f}, loss G: {g_loss[-1]:.4f}")
        if epoch%10 ==0 :
            gen_model.eval()
            with torch.no_grad():
                y_fake = gen_model(noise)
                y_fake = y_fake * 0.5 + 0.5  # remove normalization#
                save_image(y_fake, "./samples" + f"/y_gen_{epoch}.png")
            gen_model.train()


## main function
if __name__=="__main__":
    ### convert the data into tensor and apply normalisation
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
    #### download the EMNIST dataset(balanced type that include 47 classes)
    train_data = dataset.EMNIST(root='data/train_data/',split="balanced",train=True,download=True,transform=transform)
    # test_data =  dataset.EMNIST(root="data/test_data/",split="balanced",train=False,download=True)
    # print(train_data) ## the size of train dataset is 112800 (28X28)
    # print(test_data)
    ### dataloader
    train_dataloader = DataLoader(train_data,batch_size=config.batch_size,shuffle=True,num_workers=0,pin_memory=2)

    torch.manual_seed = 32
    gen_model=Generator()
    gen_model=gen_model.to(config.DEVICE)
    dis_model=Discriminator(1)
    dis_model=dis_model.to(config.DEVICE)
    fixed_noise = torch.randn((config.batch_size, 100)).to(config.DEVICE)
    gen_model.apply(initialize_weights)
    dis_model.apply(initialize_weights)
    train(train_dataloader,gen_model,dis_model,config.batch_size,config.epochs,config.gen_learning_rate,config.dis_learning_rate)

