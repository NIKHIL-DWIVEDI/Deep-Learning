import pandas as pd
import numpy as np
import torch
from zipfile import ZipFile
import torchvision
import glob
import os

## extract the zip file
# with ZipFile("archive.zip",'r') as zip:
#     zip.extractall()

# train_data = torchvision.datasets.ImageFolder(root="")
# ls = glob.glob("C:/Users/HP/Documents/Tution Classes/CNN/tiny-imagenet-200/train/*")
ls= os.listdir("C:/Users/HP/Documents/Tution Classes/CNN/tiny-imagenet-200/train")
for i in range(len(ls)):
    ls[i]=ls[i]+"/images/"
    break
print(ls[0])
# C:\Users\HP\Documents\Tution Classes\CNN\tiny-imagenet-200\train\n01644900\images\n01644900_0.JPEG
path = "C:/Users/HP/Documents/Tution Classes/CNN/tiny-imagenet-200/train/"+ls[0]+"n01443537_0.JPEG"

data=torchvision.datasets.ImageFolder(root=path)
print(data)
