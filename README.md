
# Classify-model-Pytorch-MNIST-CIFAR10-CUSTOM-DATASET
[(Back to table)](#table-of-contents)

### [2023-09-18] add model vit
### [2023-08-19] add predict.py code
you can use various network to train classify model, and l want the train and predict code as simple as possible

model includes :
1. [resnet](models/resnet.py)
   
   [Paper](https://arxiv.org/abs/1512.03385)
   
    Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
   The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.
   
![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/f01d0c15-1a97-4965-8c90-0f03d8afced9)


2. [repVGG](models/repVGG.py)
   
   [Paper](https://arxiv.org/abs/2101.03697)
   
      We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.
   
 ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/d69650c7-4279-4f12-8b74-9fa12bbea749)


3. [res2net](models/res2net.py)
 
   [Paper](https://arxiv.org/abs/1904.01169)
   
   Representing features at multiple scales is of great importance for numerous vision tasks. Recent advances in backbone convolutional neural networks (CNNs) continually demonstrate stronger multi-scale representation ability, leading to consistent performance gains on a wide range of applications. However, most existing methods represent the multi-scale features in a layer-wise manner. In this paper, we propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models on widely-used datasets, e.g., CIFAR-100 and ImageNet. Further ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, further verify the superiority of the Res2Net over the state-of-the-art baseline methods. The source code and trained models are available on this https URL.

 ![image](https://github.com/cuteboyqq/Simple-Classify-Pytorch/assets/58428559/16d34626-84b0-450b-a6be-1ec9f140058f)

   
4. [VGG16](models/VGG16.py)
   
   [Paper](https://arxiv.org/abs/1409.1556)
   
   In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.


8. 
9. [shufflenet](models/shufflenet.py)  (2022/09/18 train have error)
10. [EfficientNet](models/efficientnet.py) 
11. [MobileNet](models/mobilenet.py) (2022/09/18 train have error)
12. [MobileNetV2](models/mobilenetv2.py)
13. [LeNet](models/lenet.py)
14. [DenseNet](models/densenet.py)
15. [ShuffleNetV2](models/shufflenetv2.py)
16. [Vit](models/vit.py)
17. [Cait](models/cait.py)
18. [cct](models/cct.py)
19. [simple-vit](models/simple_vit.py)

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


### Create-Environment
[(Back to top)](#Classify-model-Pytorch-MNIST-CIFAR10-CUSTOMDATASET)

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
[(Back to top)](#Classify-model-Pytorch-MNIST-CIFAR10-CUSTOMDATASET)

```
pip install -r requirement.txt
```


### Usage
[(Back to top)](#Classify-model-Pytorch-MNIST-CIFAR10-CUSTOMDATASET)


### Train
[(Back to top)](#Classify-model-Pytorch-MNIST-CIFAR10-CUSTOMDATASET)

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
[(Back to top)](#Classify-model-Pytorch-MNIST-CIFAR10-CUSTOMDATASET)

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
