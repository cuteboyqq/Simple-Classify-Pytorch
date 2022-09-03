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
from models.resnet import ResNet,ResBlock
from models.repVGG import RepVGG,RepVGGBlock
from models.res2net import Res2Net,Bottle2neck
from models.VGG16 import VGG16
from tqdm import tqdm
from util.colorstr import colorstr
from util.data_loader import load_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''loss function'''
criterion = nn.CrossEntropyLoss()

def main(opts):
    torch.cuda.empty_cache()
    if opts.data=='mnist':
        nc=1
    elif opts.data=='cifar10':
        nc=3
    else:
        nc=opts.nc
        
    '''load Convolution Neural Network'''
    if opts.model=='resnet' or opts.model=='Resnet' or opts.model=='ResNet':
        model = ResNet(ResBlock,nc=nc)
    elif opts.model=='repvgg' or opts.model=='RepVGG' or opts.model=='Repvgg' or opts.model=='RepVgg':
        model = RepVGG(num_classes=10)
    elif opts.model=='vgg16' or opts.model=='VGG16' or opts.model=='Vgg16':
        model = VGG16()
    elif opts.model=='res2net' or opts.model=='Res2net' or opts.model=='Res2Net':
        model = Res2Net()  
    
    '''load data'''
    train_loader,test_loader = load_data(opts)
    print('model :{}'.format(opts.model))
    '''Start train/test epochs and save best.pt model'''
    train(model,train_loader,test_loader,opts)
    
    
    
def train(model,train_loader,test_loader,opts):
    if torch.cuda.is_available():
        model.cuda() 
    ''' optimizer method '''
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    _lowest_loss = 1000.0
    SAVE_MODEL_DIR = r".\runs\train"
    SAVE_MODEL_PATH = r".\runs\train\best.pt"
    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)
        
    for epoch in range(opts.epoch):
        print('{}{:7}{}{:7}{}{:7}{}{:7}{}{:7}{}'.format('Epoch','','Total_loss','','loss','','acc','','img_size','','batch_size'))
        print('--------------------------------------------------------------------------------------')
        tot_loss = 0.0
        pbar = tqdm(train_loader) #show bar progress
        for i, (inputs, labels) in enumerate(pbar):
            '''get batch images and corresponding labels'''
            inputs, labels = inputs.to(device), labels.to(device)
            '''initial optimizer to zeros'''
            optimizer.zero_grad()
            ''' put batch images to convolution neural network '''
            outputs = model(inputs)
            """calculate loss by loss function"""
            loss = criterion(outputs, labels)
            '''after calculate loss, do back propogation'''
            loss.backward()
            '''optimize weight and bais'''
            optimizer.step()
            tot_loss += loss.data   
            '''show pbar messages'''
            bar_str =   ' '+ "{0:.3f}".format(epoch)\
                          + '         ' + "{0:.3f}".format(tot_loss)\
                          + '         ' + "{0:.3f}".format(loss)\
                          + '         ' \
                          + '         ' + "{}".format(opts.img_size)\
                          + '         ' + "{}".format(opts.batch_size)
            PREFIX = colorstr(bar_str)
            pbar.desc = f'{PREFIX}'                 
        if tot_loss < _lowest_loss:
            _lowest_loss = tot_loss
            #print('Start save model !')
            torch.save(model, SAVE_MODEL_PATH)
            #print('save model complete with loss : %.3f' %(tot_loss))  
        '''test at each epochs'''
        test(model,test_loader)


def test(model,test_loader):
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
            bar_str ='{:42}'.format('')+"{0:.3f}".format(acc)
            PREFIX = colorstr(bar_str)
            pbar_test.desc = f'{PREFIX}'
            
        #print('Test\'s ac is: %.3f%%' % (100 * correct / total))

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-data','--data',help='train data (mnist, cifar10, custom data)',default=r'cifar10')
    parser.add_argument('-datatest','--data-test',help='custom test data)',default=r'C:\GitHub_Code\cuteboyqq\TLR\datasets\roi-test')
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=32)
    parser.add_argument('-nc','--nc',type=int,help='num of channels',default=3)
    parser.add_argument('-batchsize','--batch-size',type=int,help='batch-size',default=64)
    parser.add_argument('-epoch','--epoch',type=int,help='num of epochs',default=30)
    parser.add_argument('-model','--model',help='resnet,VGG16,repvgg,res2net',default='resnet')
    return parser.parse_args()    

if __name__ == "__main__":
    opts = get_args()
    main(opts)

            