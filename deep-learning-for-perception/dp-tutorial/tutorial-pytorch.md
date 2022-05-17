# Tutorial: PyTorch

## Tutorial: PyTorch

## PyTorch

### Installation

1.  Install PyTorch for Cuda 10.2

    Run the command in Anaconda Prompt(administrator mode) Python 3.7 Environment. [See here for more detail](https://ykkim.gitbook.io/dlip/dlip-installation-guide/framework/pytorch)

> Youn should install pytorch in a virtual environment

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```



2\. Watch Video of [Introduction to PyTorch (20min)](https://youtu.be/IC0\_FRiX-sw)

* You need to know 'What is Tensor in Pytorch'.

3\. Follow Quick-Start Tutorial: [Pytorch Tutorial](https://tutorials.pytorch.kr/beginner/basics/quickstart\_tutorial.html)

* Finish this tutorial before class



## Class Tutorial

### MLP

* T1-1: [Train a simple MLP (MNIST)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_PyTorch\_MNIST\_MLP\_Part1\_Train.ipynb)
* T1-2: [Test with loading trained model](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_PyTorch\_MNIST\_MLP\_Part2\_Test.ipynb)

###

### CNN- Classification

#### **Create a simple CNN model**

Download module:  [My\_DLIP.py](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/MY\_DLIP.py)

* T2-1: [Create LeNeT CNN model and Train with opendataset (CIFAR10)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Tutorial\_PyTorch\_LeNet5\_CIFAR10\_CNN\_Part1.ipynb)
* T2-2: [Test with loading trained model(LeNet-5)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Tutorial\_PyTorch\_LeNet5\_CIFAR10\_CNN\_Part2.ipynb)
* T2-3-1: [Create a CNN model(VGG-16) for ImageNet](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Tutorial\_PyTorch\_VGG16\_CNN\_Part3\_1.ipynb)
* T2-3-2: [Create, Train and Test a CNN model(VGG-16) for CIFAR10](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Tutorial\_PyTorch\_VGG16\_CIFAR10\_CNN\_Part3\_2.ipynb)

#### **Using Popular CNN models from torchvision.models**

* T3-1: [Test using Pretrained Model (VGG, Inception, ResNet)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Tutorial\_PyTorch\_T3\_1\_Inference\_using\_Pre\_trained\_Model\_\(classification\).ipynb)
* T3-2: [Train Opendataset with Transfer Learning of Pretrained model](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Tutorial\_PyTorch\_T3\_2\_Transfer\_Learning\_using\_Pre\_trained\_Models\_\(classification\).ipynb)

#### Assignment: Classification&#x20;

* T3-3: [(Assignment) Classification with Custom Dataset](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Assignment\_PyTorch\_T3\_3\_Transfer\_Learning\_using\_Pre\_trained\_Models\_\(classification\).ipynb)

###

### CNN- Object Detection

#### **YOLO v5 in PyTorch**

* T4-1: [Test using Pretrained YOLOv5 with COCO dataset](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Tutorial\_PyTorch\_T4\_1\_Inference\_using\_Pretrained\_YOLOv5\_with\_COCO\_2022.ipynb)
* T4-2: Train YOLOv5 with a Custom Dataset

## LAB

### LAB: Object Detection 1&#x20;

Parking Vehicle Detection&#x20;



### LAB: Object Detection 2 (Custom data)

You should design an object detection problem related to your interested area. For the dataset, you can search kaggle.com or any other open dataset sites.

###

## Useful Sites

1. Pytorch tutorial codes: [Pytorch-Tutorial](https://github.com/yunjey/pytorch-tutorial)
