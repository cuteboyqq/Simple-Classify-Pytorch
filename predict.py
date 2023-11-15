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
from util.data_loader import load_data,load_data_predict
import glob
from PIL import Image
import torchvision.transforms as transforms
import shutil
import numpy as np
import cv2
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-datapredict','--data-predict',help='custom test data)',\
                        default=r'/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/val')#'/home/ali/Projects/datasets/BDD100K_Val_crop/val' #'/home/ali/Projects/datasets/snow_crop_2cls/'
                        #driver_161_90frame_crop_2cls
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=128)

    parser.add_argument('-imgw','--img-w',type=int,help='image size w',default=640)
    parser.add_argument('-imgh','--img-h',type=int,help='image size h',default=64)

    parser.add_argument('-nc','--nc',type=int,help='num of channels',default=3)
    parser.add_argument('-model','--model',help='resnet,VGG16,repvgg,res2net',default='resnet')
    parser.add_argument('-mpath','--model-path',help='pretrained model path',\
                        default=r'/home/ali/Projects/GitHub_Code/ali/Simple-Classify-Pytorch/runs/train/CULane/resnet_best.pt')
    parser.add_argument('-numcls','--num-cls',type=int,help='num of class',default=2)
    return parser.parse_args()

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opts = get_args()

predict_loader = load_data_predict(opts)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''loss function'''
criterion = nn.CrossEntropyLoss()

'''   Load specific model (ex: resnet, repVGG,etcs1)'''
model = load_model(opts,opts.nc) #For example : model = ResNet(ResBlock,nc=nc)
print('model :{}'.format(opts.model))
#print(model)

#model.load_state_dict(torch.load(opts.model_path)) #load pre-trained model
model = torch.load(opts.model_path)

if torch.cuda.is_available():
    #print("cuda is available")
    model.cuda()

if opts.img_h is not None and opts.img_w is not None:
    size = (opts.img_h,opts.img_w)
else:
    size = opts.img_size
pre_process = transforms.Compose([
                transforms.Resize(size),
                #transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size),
                transforms.ToTensor()
                #normalize
                ])

os.makedirs(r".\runs\model_predict", exist_ok=True)

_lowest_loss = 1000.0
#--------------------------------------------------------------------------------------------
def parse_path(path):
    dir = os.path.dirname(path)
    label = os.path.basename(dir)
    return label
#------------------------------------------------------------------------------------------------------------
#study how to imfer one image https://medium.com/@myravithar/deriving-inference-for-new-images-using-pretrained-models-in-pytorch-8e294351c5a4
def predict():
    #get the ac with testdataset in each epoch
    #print('Waiting Test...')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    predict_img_list = glob.glob(os.path.join(opts.data_predict,"**/*.jpg"))
    class_dict={0:"0",1:"1"}
    os.makedirs("./runs/predict",exist_ok=True)
    for i in class_dict:
        os.makedirs("./runs/predict/"+class_dict[i],exist_ok=True)

    correct = 0
    wrong = 0
    total = 0
    lm = 0
    ss = 0
    ot = 0
    # acc_dict = {"lanemarking":0,"others":0,"stopsign":0}
    # total_dict = {"lanemarking":0,"others":0,"stopsign":0}
    acc_dict = {"0":0,"1":0}
    FP_dict = {"0":0,"1":0}
    total_dict = {"0":0,"1":0}
    list_bar = tqdm(predict_img_list)
    with torch.no_grad():
        for pred_img_path in list_bar:
            model.eval()
            img =  Image.open(pred_img_path)# READ IMAGE
            label = parse_path(pred_img_path)
            #img = img.to(device)
            # DEFINE TRANSFORMS
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                 std=[0.229, 0.224, 0.225])
            trans_img = pre_process(img)
            #trans_img.to(device)
            if opts.img_w is not None  and opts.img_h is not None:
                trans_img = trans_img.view([1,opts.nc,opts.img_h,opts.img_w])
            else:
                trans_img = trans_img.view([1,opts.nc,opts.img_size,opts.img_size])
            #print(trans_img.shape)
            pred = model(trans_img.cuda())
            #pred = softmax(pred.cpu().numpy())
            #print(pred)
            max_value = pred.max()
            pred_cls = pred.argmax() #get the max score label
            #print(int(max_value.cpu().numpy()))

            if class_dict[int(pred_cls.cpu().numpy())]==label:
                correct+=1
                acc_dict[label] += 1
            else:
                wrong+=1
                FP_dict[label] += 1
            
            total += 1
            acc = float(correct / total)

            total_dict[label]+=1
            #print("correct:{}   total")
            #if int(max_value.cpu().numpy())>=3.5:
            #print("predict result : {}".format(class_dict[int(pred_cls.cpu().numpy())]))
            os.makedirs(os.path.join("./runs/predict/",str(label),str(class_dict[int(pred_cls.cpu().numpy())])),exist_ok=True)
            shutil.copy(pred_img_path,os.path.join("./runs/predict/",str(label),str(class_dict[int(pred_cls.cpu().numpy())])))
            #else:
            #    print("max_value<=3.5")
            bar_str ='GT:{}'.format(label) + '  cor:{}'.format(correct)+'  wro:{}'.format(wrong)  +"    to1:{}".format(total) + "     acc:{0:.3f}".format(acc)
            PREFIX = colorstr(bar_str)
            list_bar.desc = f'{PREFIX}'

        print("correct count:")
        print("label                total       TP          FN    acc")
        for num,key in enumerate(acc_dict):
            print("{:15} {:10} {:10} {:10} {:20}".format(class_dict[num],\
                                                        total_dict[key],\
                                                        acc_dict[key],\
                                                        FP_dict[key],\
                                                        float(acc_dict[key]/total_dict[key])))

## Unable to save images
## try to use GPU inference
def predict_v2():
    #get the ac with testdataset in each epoch
    #print('Waiting Test...')

    class_dict={0:"landmark",1:"stopsign",2:"others"}
    os.makedirs("./runs/predict",exist_ok=True)
    for i in class_dict:
        os.makedirs("./runs/predict/"+str(i),exist_ok=True)

    with torch.no_grad():
        correct = 0
        total = 0
        tot_loss_test = 0.0
        pbar_predict = tqdm(predict_loader)
        for data in pbar_predict:
            model.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_test = criterion(outputs, labels)
            tot_loss_test += loss_test.data
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            '''save predict image'''
            print("predict result : {}".format(class_dict[int(predicted.cpu().numpy())]))
            img = images.cpu().squeeze().numpy()
            img = cv2.imread(img)
            shutil.copy(img,"./runs/predict/"+str(int(predicted.cpu().numpy())))

            '''show pbar messages'''
            acc = correct/total
            bar_str ='{:30}'.format('')+"{0:.3f}".format(acc)
            PREFIX = colorstr(bar_str)
            pbar_predict.desc = f'{PREFIX}'
            
        #print('Test\'s ac is: %.3f%%' % (100 * correct / total))

            
if __name__ == "__main__":
    predict()

            