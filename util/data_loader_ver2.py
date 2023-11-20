# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 01:49:35 2022

@author: User
"""
import torch
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import glob
import os 
from sklearn.model_selection import train_test_split
from PIL import Image

## dataset for regression labels
## If you want to use custom dataset
## Step 0 : Create your cunstom re-gression labels
## Use you custom parsing code to get re-gression label 
## for example :
#    label = ((img_path.split(os.sep)[-1]).split(".")[0]).split("-")[-1]
class dataset(torch.utils.data.Dataset):
     
    def __init__(self,file_list,transform=None):
          self.file_list = file_list
          self.transform = transform
          
    def __getitem__(self,idx):
        ## Get image and transform image
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        ## customize the regression label by yourself 
        ## (Base on your own dataset label format)
        label = ((img_path.split(os.sep)[-1]).split(".")[0]).split("-")[-1]
        label = int(label)

        return img_transformed,label
     
    def __len__(self):
        self.file_length = len(self.file_list)
        return self.file_length

def load_data(opts):
    batch_size = opts.batch_size
    if not opts.img_w==None and not opts.img_h==None:
        size = (opts.img_h,opts.img_w)
    else:
        size = (opts.img_size,opts.img_size)

    train_list = glob.glob(os.path.join(opts.train_dir,'*.jpg'))

    if opts.train_split:
         train_list, val_list = train_test_split(train_list,test_size=0.2)
    else:
         val_list = glob.glob(os.path.join(opts.val_dir,'*.jpg'))

    ## Create data transform
    train_transform=transforms.Compose([
                                    transforms.Resize(size),
                                    transforms.CenterCrop(size),
                                    transforms.ToTensor()
                                    ])
    
    val_transform=transforms.Compose([
                                    transforms.Resize(size),
                                    transforms.CenterCrop(size),
                                    transforms.ToTensor()
                                    ])
    
    ## Create data by calling custom class dataset
    train_data = dataset(train_list,train_transform)
    val_data = dataset(val_list,val_transform)

    ## Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True,drop_last=False)
    val_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=False,drop_last=False)

    return train_loader,val_loader


def load_data_predict(opts):
    if not opts.img_w==None and not opts.img_h==None:
            size = (opts.img_h,opts.img_w)
    else:
        size = (opts.img_size,opts.img_size)

    predict_data = torchvision.datasets.ImageFolder(opts.data_predict,
                                                transform=transforms.Compose([
                                                    transforms.Resize(size),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.Scale(64),
                                                    transforms.CenterCrop(size),
                                                    
                                                    transforms.ToTensor()
                                                    ])
                                                )
    
    
    predict_loader  = torch.utils.data.DataLoader(predict_data, batch_size=1,shuffle=False,drop_last=False)
    
    return predict_loader