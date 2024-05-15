# Tutorial: PyTorch Tutorial List

## Preparation for PyTorch Tutorial

### PyTorch Installation

#### Install PyTorch

Follow the installatin instruction: [See here for more detail](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning#part-3.-installing-dl-framework)

> You should install pytorch in a virtual environment

### Check PyTorch Installation and GPU

In the Anaconda Promt, type

```cpp
conda activate py39
python
import torch
torch.__version__
print("cuda" if torch.cuda.is_available() else "cpu")
```



### Watch PyTorch Intro  Video

[Introduction to PyTorch (20min)](https://youtu.be/IC0\_FRiX-sw)

> You need to know 'What is Tensor in Pytorch'

### **Follow Quick-Start Tutorial:**

* [Pytorch Tutorial(ENG)](https://pytorch.org/tutorials/beginner/basics/quickstart\_tutorial.html)
* [Pytorch Tutorial(KOR)](https://tutorials.pytorch.kr/beginner/basics/quickstart\_tutorial.html)

***

## DLIP Course Tutorials

### MLP

* T1: [Train a simple MLP and Test with loading trained model (MNIST)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T1_MNIST_MLP_2024.ipynb)

### CNN- Classification

#### **Create a simple CNN model**

* T2-1: [Create LeNeT CNN model, Train with opendataset, and Test with loading trained model (CIFAR10)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_1_LeNet5_CIFAR10_CNN_2024.ipynb)
* T2-2-1: [Create a CNN model(VGG-16) for ImageNet](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_2_1_VGG16_CNN_2024.ipynb)
* T2-2-2: [Create, Train and Test a CNN model(VGG-16) for CIFAR10](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_2_2_VGG16_CIFAR10_CNN_2024.ipynb)

#### **Using Popular CNN models from torchvision.models**

* T3-1: [Test using Pretrained Model (VGG, Inception, ResNet)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T3_1_Inference_using_Pre_trained_Model_(classification)_2024.ipynb)
* T3-2: [Train Opendataset with Transfer Learning of Pretrained model](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T3_2_Transfer_Learning_using_Pre_trained_Models_(classification)_2024.ipynb)

#### Assignment: Classification

* T3-3: [(Assignment) Classification with Custom Dataset](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Assignment_PyTorch_T3_3_Transfer_Learning_using_Pre_trained_Models_(classification)_2024.ipynb)
* T3-4: [(Assignment) Create ResNet-50 model for ImageNet](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Assignment_PyTorch_T3_4_ResNet50_2024.ipynb)

### CNN- Object Detection

#### **YOLO v8 in PyTorch**

* T4-1: [Install and Inference using  YOLOv8 ](tutorial-yolov8-in-pytorch.md#tutorial-yolo-v8-in-pytorch)
* T4-2: [Train and Test using Custom Dataset ](tutorial-yolov8-in-pytorch.md#tutorial-yolo-v8-in-pytorch)

#### **YOLO v5 in PyTorch**

* T4-1(option1): [Pretrained YOLOv5 with COCO dataset](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/Tutorial\_PyTorch\_T4\_1\_Inference\_using\_Pretrained\_YOLOv5\_with\_COCO\_2022.ipynb) (in CoLab)
* T4-1(option2):[ Pretrained YOLOv5 with COCO dataset](tutorial-yolov5-in-pytorch/) (in VS Code, Local PC)
* T4-2: [Train YOLOv5 with a Custom Dataset](tutorial-yolov5-in-pytorch/tutorial-yolov5-train-with-custum-data.md) (in VS Code, Local PC)



***

### Useful Sites

Pytorch tutorial codes: [Pytorch-Tutorial](https://github.com/yunjey/pytorch-tutorial)&#x20;

Pytorch Tutorial List: [PyTorch Tutorial List](../../programming/pytorch/)
