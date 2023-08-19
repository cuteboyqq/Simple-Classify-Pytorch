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
import glob
from PIL import Image
import torchvision.transforms as transforms
import shutil


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-datatest','--data-test',help='custom test data)',default=r'/home/jnr_loganvo/Alister/datasets/landmark')
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=64)
    parser.add_argument('-nc','--nc',type=int,help='num of channels',default=3)
    parser.add_argument('-model','--model',help='resnet,VGG16,repvgg,res2net',default='resnet')
    parser.add_argument('-mpath','--model-path',help='pretrained model path',default=r'/home/jnr_loganvo/Alister/GitHub_Code/Simple-Classify-Pytorch/runs/train/resnet_best.pt')
    return parser.parse_args()

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opts = get_args()

'''   Load specific model (ex: resnet, repVGG,etcs1)'''
#model = load_model(opts,opts.nc) #For example : model = ResNet(ResBlock,nc=nc)
#print('model :{}'.format(opts.model))

#model.load_state_dict(torch.load(opts.model_path)) #load pre-trained model
model = torch.load(opts.model_path)

if torch.cuda.is_available():
    model.cuda()

pre_process = transforms.Compose([
                transforms.Resize(opts.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(opts.img_size),
                transforms.ToTensor()
                #normalize
                ])

os.makedirs(r".\runs\predict", exist_ok=True)

_lowest_loss = 1000.0
#--------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
#study how to imfer one image https://medium.com/@myravithar/deriving-inference-for-new-images-using-pretrained-models-in-pytorch-8e294351c5a4
def predict():
    #get the ac with testdataset in each epoch
    #print('Waiting Test...')
    predict_img_list = glob.glob(os.path.join(opts.data_test,"*/*.jpg"))
    class_dict={0:"landmark",1:"others"}
    os.makedirs("./runs/predict",exist_ok=True)
    for i in class_dict:
        os.makedirs("./runs/predict/"+str(i),exist_ok=True)

    with torch.no_grad():
        for pred_img_path in predict_img_list:
            model.eval()
            img =  Image.open(pred_img_path)# READ IMAGE
            # DEFINE TRANSFORMS
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                 std=[0.229, 0.224, 0.225])
            trans_img = pre_process(img)
            trans_img = trans_img.view([1,opts.nc,opts.img_size,opts.img_size])
            pred = model(trans_img)
            pred_cls = pred.argmax() #get the max score label
            print("predict result : {}".format(class_dict[int(pred_cls.numpy())]))
            shutil.copy(pred_img_path,"./runs/predict/"+str(int(pred_cls.numpy())))
            
            
if __name__ == "__main__":
    predict()

            