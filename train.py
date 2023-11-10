# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 22:58:24 2022

@author: User
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import torch.utils.data
from torch import nn, optim
from util.load_model import load_model
from tqdm import tqdm
from util.colorstr import colorstr
from util.data_loader import load_data



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-data','--data',help='train data (mnist, cifar10, or custom data directory)',default=r'/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train')
    parser.add_argument('-datatest','--data-test',help='custom test data)',default=r'/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/val')
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=64)
    parser.add_argument('-imgw','--img-w',type=int,help='image width',default=64)
    parser.add_argument('-imgh','--img-h',type=int,help='image height',default=640)
    parser.add_argument('-nc','--nc',type=int,help='num of channels',default=3)
    parser.add_argument('-batchsize','--batch-size',type=int,help='batch-size',default=64)
    parser.add_argument('-epoch','--epoch',type=int,help='num of epochs',default=50)
    parser.add_argument('-model','--model',help='resnet,VGG16,repvgg,res2net',default='resnet')
    return parser.parse_args()

torch.cuda.empty_cache()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''loss function'''
criterion = nn.CrossEntropyLoss()

opts = get_args()
''' Load data (ex:mnist, cifar10, or custom dataset)'''
train_loader,test_loader = load_data(opts)


if opts.data=='mnist':
    nc=1
elif opts.data=='cifar10':
    nc=3
else:
    nc=opts.nc
    
'''   Load specific model (ex: resnet, repVGG,etcs1)'''
model = load_model(opts,nc)
print('model :{}'.format(opts.model))

if torch.cuda.is_available():
    model.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

os.makedirs(r"./runs/train", exist_ok=True)

_lowest_loss = 1000.0
#--------------------------------------------------------------------------------------------
def train(epoch):
    global _lowest_loss
    #for epoch in range(opts.epoch):
    print('{}{:4}{}{:4}{}{:4}{}{:4}{}{:4}{}{:4}{}{:4}{}'.format('Epoch','','Total_loss','','loss','','acc','','img_size','','bs','','model','','data'))
    #print('-----------------------------------------------------------------------------')
    tot_loss = 0.0
    pbar = tqdm(train_loader) #show bar progress
    for i, (inputs, labels) in enumerate(pbar):              
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        tot_loss += loss.data   
        '''show pbar messages'''
        if opts.img_w is not None and opts.img_h is not None:
            img_size = (opts.img_h,opts.img_w)
        else:
            img_size = opts.img_size
        bar_str =   ' '+ "{0:.3f}".format(epoch)\
                      + '      ' + "{0:.3f}".format(tot_loss)\
                      + '      ' + "{0:.3f}".format(loss)\
                      + '      ' \
                      + '      ' + "{}".format(img_size)\
                      + '      ' + "{}".format(opts.batch_size)\
                      + '      ' + "{}".format(opts.model)\
                      + '      ' + "{}".format(opts.data)
        PREFIX = colorstr(bar_str)
        pbar.desc = f'{PREFIX}'                 
    if tot_loss < _lowest_loss:
        _lowest_loss = tot_loss
        model_name =  opts.model + "_best.pt"
        torch.save(model, os.path.join(r"./runs/train",model_name))
#------------------------------------------------------------------------------------------------------------
def test():
    #get the ac with testdataset in each epoch
    #print('Waiting Test...')
    pbar_test = tqdm(test_loader)
    with torch.no_grad():
        correct = 0
        total = 0
        tot_loss_test = 0.0
        for data in pbar_test:
            model.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_test = criterion(outputs, labels)
            tot_loss_test += loss_test.data
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            '''show pbar messages'''
            acc = correct/total
            bar_str ='{:30}'.format('')+"{0:.3f}".format(acc)
            PREFIX = colorstr(bar_str)
            pbar_test.desc = f'{PREFIX}'
            
        #print('Test\'s ac is: %.3f%%' % (100 * correct / total))
if __name__ == "__main__":
    for epoch in range(opts.epoch):
        train(epoch)
        test()

            