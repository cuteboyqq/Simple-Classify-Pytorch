
# Classify-model-Pytorch-MNIST-CIFAR10-CUSTOM-DATASET


### Model-History
[(Back-to-Table)](#table-of-contents)

1. [Sim-Vit](#Sim-Vit) [Network](models/simple_vit.py) Author : G Li  · 2021 · Referenced : 10 
1. [Cait](#Cait) [Network](models/cait.py) Author : H Touvron  · 2021 · referenced : 728
2. [Vit](#Vit) [Network](models/vit.py) Author : A Dosovitskiy  · 2020 · Referenced : 24385
3. [RepVGG](#RepVGG) [Network](models/repVGG.py) Author : X Ding  · 2021 · Referenced : 1010
4. [Res2Net](#Res2Net) [Network](models/res2net.py) Author : SH Gao  · 2019 · Referenced : 1998
5. [EfficientNet](#EfficientNet) [Network](models/efficientnet.py) Author : M Tan  · 2019 · Referenced : 15999
6. [MobileNetV2](#MobileNetV2) [Network](models/mobilenetv2.py) Author : M Sandler  · 2018 · Referenced : 18573
7. [ShuffleNetV2](#ShuffleNetV2) [Network](models/shufflenetv2.py) Author : N Ma · 2018 · Referenced : 4622  
8. [MobileNet](#MobileNet)  [Network](models/mobilenet.py) Author : AG Howard  · 2017 · Referenced : 22147
9. [ShuffleNet](#ShuffleNet) [Network](models/shufflenet.py) Author : X Zhang  · 2017 · Referenced : 6972
10. [ResNet](#ResNet) [Network](models/resnet.py) Author : K He  · 2015 · Referenced : 188186
11. [VGG16](#VGG16) [Network](models/VGG16.py) Author : K Simonyan  · 2014 · Referenced : 112804


<!-- Add a demo for your project -->

<!-- After you have written about your project, it is a good idea to have a demo/preview(**video/gif/screenshots** are good options) of your project so that people can know what to expect in your project. You could also add the demo in the previous section with the product description.

Here is a random GIF as a placeholder.

![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) -->

### table-of-contents
- [Classify-model-Pytorch](#Classify-model-Pytorch-MNIST-CIFAR10-CUSTOM-DATASET)
- [Create-Environment](#Create-Environment)
- [Requirement](#Requirement)
- [Usage](#usage)
    - [Train](#Train)
    - [Predict](#Predict)
- [Model-History](#Model-History)
- [Model-Abstract](#Model-Abstract)


[(Back to table)](#table-of-contents)

### [2023-09-18] add model vit
### [2023-08-19] add predict.py code
you can use various network to train classify model, and l want the train and predict code as simple as possible

### Create-Environment
[(Back to top)](#table-of-contents)

```
conda create --name Classify
```
```
conda activate Classify
```
```
git clone https://github.com/cuteboyqq/Simple-Classify-Pytorch.git
```
### Requirement
[(Back to top)](#table-of-contents)

```
pip install -r requirement.txt
```

### Usage
[(Back to top)](#table-of-contents)


### Train
[(Back to top)](#table-of-contents)

user can train classify model by cifar/mnist/custom datasets, command is below
```
python train.py --data [Enter open datasets name(Ex:mnist,cifar10) or custom dataset directory] 
                --img-size [Enter image size (Ex:32 or 64 or 128)] 
                --batch-size [Enter batch size (Ex:32 or 64 or 128)] 
                --nc [Enter number of channels (Ex:1 or 3)] 
                --model [Enter model name (Ex:resnet,mobilenetv2,shufflenetv2,etcs)]
```
For examples :
```
python train.py --data cifar10 --img-size 32 --batch-size 64 --nc 3 --model resnet
```
```
python train.py --data mnist --img-size 32 --batch-size 64  --model ShuffleNetV2
```

```
python train.py --data [Enter custom dataset directory] --img-size 32 --batch-size 64  --model  MobileNetV2
```
### Predict
[(Back to top)](#table-of-contents)

after training done, you will have model at folder ./runs/train/ , so use this model to predict images and will save predict result images
```
python predict.py --data-test [predict data directory] 
                --img-size [Enter image size (Ex:32 or 64 or 128)] 
                --nc [Enter number of channels (Ex:1 or 3)] 
                --model [Enter model name (Ex:resnet,mobilenetv2,shufflenetv2,etcs)]
                --model-path [Enter the path of model.pt(Ex:/path/to/your/runs/train/resnet_best.pt)]
```
For examples :
```
python predict.py --data cifar10 --img-size 32 --nc 3 --model [/path/to/your/runs/train/resnet_best.pt]
```


### Model-Abstract
#### ResNet
1. [Network](models/resnet.py) [Paper](https://arxiv.org/abs/1512.03385) [(Back to table)](#Model-History)

    Author : K He  · 2015 · Referenced : 188186 
   
    Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
   The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.
   
   ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/f01d0c15-1a97-4965-8c90-0f03d8afced9)

#### RepVGG
2. [Network](models/repVGG.py)  [Paper](https://arxiv.org/abs/2101.03697) [(Back to table)](#Model-History)

      Author : X Ding  · 2021 · Referenced : 1010
      
      We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.
   
   ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/cfa08dc5-58f1-4031-b95a-c51cb5983227)

#### Res2Net
4. [Network](models/res2net.py) [Paper](https://arxiv.org/abs/1904.01169) [(Back to table)](#Model-History)

    Author : SH Gao  · 2019 · Referenced : 1998
   
   Representing features at multiple scales is of great importance for numerous vision tasks. Recent advances in backbone convolutional neural networks (CNNs) continually demonstrate stronger multi-scale representation ability, leading to consistent performance gains on a wide range of applications. However, most existing methods represent the multi-scale features in a layer-wise manner. In this paper, we propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models on widely-used datasets, e.g., CIFAR-100 and ImageNet. Further ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, further verify the superiority of the Res2Net over the state-of-the-art baseline methods. The source code and trained models are available on this https URL.

   ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/5527ca99-c6b0-45f5-b2b2-8b577b5cb8ae)

#### VGG16
4. [Network](models/VGG16.py) [Paper](https://arxiv.org/abs/1409.1556) [(Back to table)](#Model-History)

   Author : K Simonyan  · 2014 · Referenced : 112804
   
   In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

#### ShuffleNet
5. [Network](models/shufflenet.py) [Paper](https://arxiv.org/abs/1707.01083) [(Back to table)](#Model-History) (2022/09/18 train have error)

   Author : X Zhang  · 2017 · Referenced : 6972

   We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially
for mobile devices with very limited computing power (e.g.,10-150 MFLOPs). The new architecture utilizes two new
operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining
accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1
error (absolute 7.8%) than recent MobileNet [12] on ImageNet classification task, under the computation budget of
40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves ∼13× actual speedup over AlexNet while maintaining comparable accuracy.

   ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/f6a8a7a9-2ebe-4bd3-a455-a1bf5afc9a60)

#### EfficientNet
6. [Network](models/efficientnet.py) [Paper](https://arxiv.org/abs/1905.11946) [(Back to table)](#Model-History)

    Author : M Tan  · 2019 · Referenced : 15999
    
    Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.
To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at this https URL.

   ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/8adc23a7-6be4-4c64-8a0c-49395e958d57)



#### MobileNet
7. [Network](models/mobilenet.py) [Paper](https://arxiv.org/abs/1704.04861) [(Back to table)](#Model-History) (2022/09/18 train have error)

   Author : AG Howard  · 2017 · Referenced : 22147 
   
   We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.

   ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/abcddb1f-9661-4c75-b19f-7c43451342ea)

#### MobileNetV2
8. [Network](models/mobilenetv2.py) [Paper](https://arxiv.org/abs/1801.04381) [(Back to table)](#Model-History)

    Author : M Sandler  · 2018 · Referenced : 18573  
    
    In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3.
The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters

    ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/945a691b-113b-4752-bf81-ec68a0c953d4)


    
16. [LeNet](models/lenet.py)
17. [DenseNet](models/densenet.py)

#### ShuffleNetV2
19. [Network](models/shufflenetv2.py) [Paper](https://arxiv.org/abs/1807.11164)

    Author : N Ma · 2018 · Referenced : 4622  
    
    Currently, the neural network architecture design is mostly guided by the indirect metric of computation complexity, i.e., FLOPs.
However, the direct metric, e.g., speed, also depends on the other factors such as memory access cost and platform characterics. Thus, this work
proposes to evaluate the direct metric on the target platform, beyond only considering FLOPs. Based on a series of controlled experiments,
this work derives several practical guidelines for efficient network design. Accordingly, a new architecture is presented, called ShuffleNet V2.
Comprehensive ablation experiments verify that our model is the stateof-the-art in terms of speed and accuracy tradeoff.

    ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/7dcfea9f-3442-46df-bc6c-07ec8d586b59)


#### Vit
20. [Network](models/vit.py) [Paper](https://arxiv.org/abs/2010.11929) [(Back to table)](#Model-History)

    Author : A Dosovitskiy  · 2020 · Referenced : 24385 

    While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

    ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/15bd4576-df2f-43e1-93a6-b8db49006415)


#### Cait
21. [Network](models/cait.py) [Paper](https://arxiv.org/abs/2103.17239) [(Back to table)](#Model-History)

    Author : H Touvron  · 2021 · referenced : 728 

    Transformers have been recently adapted for large scale image classification, achieving high scores shaking up the long supremacy of convolutional neural networks. However the optimization of image transformers has been little studied so far. In this work, we build and optimize deeper transformer networks for image classification. In particular, we investigate the interplay of architecture and optimization of such dedicated transformers. We make two transformers architecture changes that significantly improve the accuracy of deep transformers. This leads us to produce models whose performance does not saturate early with more depth, for instance we obtain 86.5% top-1 accuracy on Imagenet when training with no external data, we thus attain the current SOTA with less FLOPs and parameters. Moreover, our best model establishes the new state of the art on Imagenet with Reassessed labels and Imagenet-V2 / match frequency, in the setting with no additional training data. We share our code and models.

    ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/3766457e-8d94-4b85-93af-ad9d7cf092c2)

    
24. [cct](models/cct.py)

#### Sim-Vit
26. [simple-vit](models/simple_vit.py) [Paper](https://arxiv.org/abs/2112.13085) [(Back to table)](#Model-History)

    Author : G Li  · 2021 · Referenced : 10 
    
    Although vision Transformers have achieved excellent performance as backbone models in many vision tasks, most of them intend to capture global relations of all tokens in an image or a window, which disrupts the inherent spatial and local correlations between patches in 2D structure. In this paper, we introduce a simple vision Transformer named SimViT, to incorporate spatial structure and local information into the vision Transformers. Specifically, we introduce Multi-head Central Self-Attention(MCSA) instead of conventional Multi-head Self-Attention to capture highly local relations. The introduction of sliding windows facilitates the capture of spatial structure. Meanwhile, SimViT extracts multi-scale hierarchical features from different layers for dense prediction tasks. Extensive experiments show the SimViT is effective and efficient as a general-purpose backbone model for various image processing tasks. Especially, our SimViT-Micro only needs 3.3M parameters to achieve 71.1% top-1 accuracy on ImageNet-1k dataset, which is the smallest size vision Transformer model by now. Our code will be available in this https URL.

    ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/b3a3d51c-70e9-430f-90c9-66234ff10f3b)

    ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/9a9c0371-c37e-4459-a29b-fe8dbd2b21e3)





