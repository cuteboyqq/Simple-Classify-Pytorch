# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:06:04 2022

@author: User
"""
from models.resnet import ResNet,ResBlock
from models.repVGG import RepVGG,RepVGGBlock
from models.res2net import Res2Net,Bottle2neck
from models.VGG16 import VGG16
from models.shufflenet import *
from models.efficientnet import *
from models.mobilenet import *
from models.mobilenetv2 import *
from models.lenet import *
from models.densenet import *
from models.shufflenetv2 import *
from models.vit import *
from models.simple_vit import *
from models.cct import CCT
from models.cct import *
from models.na_vit import *
from models.cait import *
from linformer import Linformer

def load_model(opts,nc):
    if opts.model=='resnet' or opts.model=='Resnet' or opts.model=='ResNet':
        model = ResNet(ResBlock,nc=nc)
    elif opts.model=='repvgg' or opts.model=='RepVGG' or opts.model=='Repvgg' or opts.model=='RepVgg' or opts.model=='repVgg' or opts.model=='repVGG':
        model = RepVGG(num_classes=10)
    elif opts.model=='vgg16' or opts.model=='VGG16' or opts.model=='Vgg16':
        model = VGG16()
    elif opts.model=='res2net' or opts.model=='Res2net' or opts.model=='Res2Net':
        model = Res2Net() 
    elif opts.model=='shufflenet' or opts.model=='ShuffleNet' or opts.model=='shuffleNet':
        model = ShuffleNetG2()
    elif opts.model=='EfficientNet' or opts.model=='efficientNet' or opts.model=='efficientnet':
        model = EfficientNetB0()
    elif opts.model=='MobileNet' or opts.model=='Mobilenet' or opts.model=='mobilenet':
        model = MobileNet()
    elif opts.model=='MobileNetV2' or opts.model=='MobileNetv2' or opts.model=='mobileNetv2' or opts.model=='mobilenetv2':
        model = MobileNetV2()
    elif opts.model=='LeNet' or opts.model=='Lenet' or opts.model=='lenet' or opts.model=='leNet':
        model = LeNet()
    elif opts.model=='DenseNet' or opts.model=='Densenet' or opts.model=='denseNet' or opts.model=='densenet':
        model = densenet_cifar()
    elif opts.model=='ShuffleNetV2' or opts.model=='shuffleNetV2' or opts.model=='shufflenetV2' or opts.model=='shufflenetv2':
        model = ShuffleNetV2(net_size=0.5)
    elif opts.model=='vit' or opts.model=='Vit' or opts.model=='VIT':
        model = ViT(
                    image_size = opts.img_size,
                    patch_size = 16,
                    num_classes = 3,
                    dim = 512,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 1024,
                    dropout = 0.1,
                    emb_dropout = 0.1
                    )
    elif opts.model=='simple-vit' or opts.model=='simple-Vit' or opts.model=='simple-VIT' or opts.model=='Simple-VIT' or opts.model=='Simple-Vit':
        model = SimpleViT(
                            image_size = 256,
                            patch_size = 32,
                            num_classes = nc,
                            dim = 1024,
                            depth = 6,
                            heads = 16,
                            mlp_dim = 2048
                        )
    elif opts.model=='cct' or opts.model=='CCT':
        model = CCT(
                    img_size = (opts.img_size,opts.img_size),
                    embedding_dim = 384,
                    n_conv_layers = 2,
                    kernel_size = 7,
                    stride = 2,
                    padding = 3,
                    pooling_kernel_size = 3,
                    pooling_stride = 2,
                    pooling_padding = 1,
                    num_layers = 14,
                    num_heads = 6,
                    mlp_ratio = 3.,
                    num_classes = 3,
                    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
                    )
    elif opts.model=='na-vit' or opts.model=='NaViT' or opts.model=='navit':
        model = NaViT(
                    image_size = opts.img_size,
                    patch_size = 32,
                    num_classes = 3,
                    dim = 512,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 1024,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
                    )
    elif opts.model=='CaiT' or opts.model=='cait' or opts.model=='CAIT':
        model = CaiT(
                    image_size = opts.img_size,
                    patch_size = 32,
                    num_classes = 3,
                    dim = 512,
                    depth = 12,             # depth of transformer for patch to patch attention only
                    cls_depth = 2,          # depth of cross attention of CLS tokens to patch
                    heads = 16,
                    mlp_dim = 1024,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    layer_dropout = 0.05    # randomly dropout 5% of the layers
                )
    return model