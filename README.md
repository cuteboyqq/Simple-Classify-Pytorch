### Classify-model-Pytorch (MNIST,CIFAR10,CUSTOM DATASET)
[(Back to top)](#table-of-contents)

model includes :
1. resnet
2. repVGG
3. res2net
4. VGG
5. shufflenet (2022/09/18 have train error)
6. EfficientNet
7. MobileNet (2022/09/18 have train error)
8. MobileNetV2
9. LeNet
10. DenseNet
11. ShuffleNetV2

<!-- Add a demo for your project -->

<!-- After you have written about your project, it is a good idea to have a demo/preview(**video/gif/screenshots** are good options) of your project so that people can know what to expect in your project. You could also add the demo in the previous section with the product description.

Here is a random GIF as a placeholder.

![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) -->

### Table of contents
- [Classify-model-Pytorch](#Classify-model-Pytorch)
- [Create-Environment](#Create-Environment)
- [Requirement](#Requirement)
- [Usage](#usage)
    - [Train](#Train)
    - [Test](#Test)


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
```
python train.py --data cifar10 --img-size 32 --batch-size 64 --nc 3
```
### Test
[(Back to top)](#table-of-contents)

not implement....maybe implement in the future
