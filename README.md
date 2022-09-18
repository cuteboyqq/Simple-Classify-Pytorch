
# Classify-model-Pytorch-MNIST-CIFAR10-CUSTOMDATASET
[(Back to table)](#table-of-contents)

model includes :
1. [resnet](models/resnet.py)
2. [repVGG](models/repVGG.py)
3. [res2net](models/res2net.py)
4. [VGG16](models/VGG16.py)
5. [shufflenet](models/shufflenet.py)  (2022/09/18 train have error)
6. [EfficientNet](models/efficientnet.py) 
7. [MobileNet](models/mobilenet.py) (2022/09/18 train have error)
8. [MobileNetV2](models/mobilenetv2.py)
9. [LeNet](models/lenet.py)
10. [DenseNet](models/densenet.py)
11. [ShuffleNetV2](models/shufflenetv2.py)

<!-- Add a demo for your project -->

<!-- After you have written about your project, it is a good idea to have a demo/preview(**video/gif/screenshots** are good options) of your project so that people can know what to expect in your project. You could also add the demo in the previous section with the product description.

Here is a random GIF as a placeholder.

![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) -->

### table-of-contents
- [Classify-model-Pytorch](#Classify-model-Pytorch-MNIST-CIFAR10-CUSTOMDATASET)
- [Create-Environment](#Create-Environment)
- [Requirement](#Requirement)
- [Usage](#usage)
    - [Train](#Train)
    - [Test](#Test)


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
```
python train.py --data [enter open datasets name(Ex:mnist,cifar10) or custom dataset directory] 
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
### Test
[(Back to top)](#Classify-model-Pytorch-MNIST-CIFAR10-CUSTOMDATASET)

not implement....maybe implement in the future
