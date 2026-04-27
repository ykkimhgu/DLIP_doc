# Tutorial: PyTorch Tutorial List

## Preparation&#x20;

### 1. Install: PyTorch&#x20;

Follow the installatin instruction: [See here for more detail](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning#part-3.-installing-dl-framework)

> You should install pytorch in a virtual environment

#### Check PyTorch Installation and GPU availability

In the Anaconda Promt, type

```cpp
conda activate py39
python
import torch
torch.__version__
print("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. Watch : Introduction Video on PyTorch&#x20;

[Introduction to PyTorch (20min)](https://youtu.be/IC0_FRiX-sw)

> You need to know 'What is Tensor in Pytorch'

### **3. Follow:  Quick-Start Tutorials**

* [Pytorch Tutorial(ENG)](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
* [Pytorch Tutorial(KOR)](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)

***

## List of Tutorials for DLIP

### 1. MLP

#### Numpy (colab)

* T0: [Numpy for a simple MLP for XOR ](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_PythonNumpy/Tutorial_Numpy_MLP_XOR_Student.ipynb)

#### PyTorch (colab)

* T1: [Train a simple MLP and Test with loading trained model (MNIST)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T1_MNIST_MLP_2024.ipynb)-colab

#### PyTorch (VSC)

* T1: [Train a simple MLP and Test with loading trained model (MNIST) in VS code](https://ykkim.gitbook.io/dlip/deep-learning-for-perception/dp-tutorial/tutorial-pytorch-example-code#example-1.-ann-mlp-model-train-test)

### 2. CNN Design- Classification

#### **Create a simple CNN model (colab)**

* T2-1: [Create LeNeT CNN model, Train with opendataset, and Test with loading trained model (CIFAR10)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_1_LeNet5_CIFAR10_CNN_2024.ipynb)
* T2-2-1: [Create a CNN model(VGG-16) for ImageNet](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_2_1_VGG16_CNN_2024.ipynb)
* T2-2-2: [Create, Train and Test a CNN model(VGG-16) for CIFAR10](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_2_2_VGG16_CIFAR10_CNN_2024.ipynb)

#### **Create a simple CNN model (VSC)**

* T2-1: [Create Train LeNeT CNN model for CIFAR10](https://ykkim.gitbook.io/dlip/deep-learning-for-perception/dp-tutorial/tutorial-pytorch-example-code#example-2.-cnn-model-train-test)

###

### 3. CNN Pretrained - Classification&#x20;

#### **Using Popular CNN models from torchvision.models**

* T3-1: [Test using Pretrained Model (VGG, Inception, ResNet)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T3_1_Inference_using_Pre_trained_Model_\(classification\)_2024.ipynb)
* T3-2: [Train Opendataset with Transfer Learning of Pretrained model](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T3_2_Transfer_Learning_using_Pre_trained_Models_\(classification\)_2024.ipynb)

#### Assignment: Classification

* T3-3: [(Assignment) Classification with Custom Dataset](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Assignment_PyTorch_T3_3_Transfer_Learning_using_Pre_trained_Models_\(classification\)_2024.ipynb)
* T3-4: [(Assignment) Create ResNet-50 model for ImageNet](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Assignment_PyTorch_T3_4_ResNet50_2024.ipynb)

###

### 4. CNN- Object Detection- YOLO

#### **YOLO v8 in PyTorch**

* T4-1: [Install and Inference using YOLOv8](tutorial-yolov8-in-pytorch.md#tutorial-yolo-v8-in-pytorch)
* T4-2: [Train and Test using Custom Dataset](tutorial-yolov8-in-pytorch.md#tutorial-yolo-v8-in-pytorch)

#### **YOLO v5 in PyTorch**

* T4-1(option1): [Pretrained YOLOv5 with COCO dataset](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/Tutorial_PyTorch_T4_1_Inference_using_Pretrained_YOLOv5_with_COCO_2022.ipynb) (in CoLab)
* T4-1(option2):[ Pretrained YOLOv5 with COCO dataset](tutorial-yolov5-in-pytorch/) (in VS Code, Local PC)
* T4-2: [Train YOLOv5 with a Custom Dataset](tutorial-yolov5-in-pytorch/tutorial-yolov5-train-with-custum-data.md) (in VS Code, Local PC)

***

### Useful Sites

Pytorch tutorial codes: [Pytorch-Tutorial](https://github.com/yunjey/pytorch-tutorial)

Pytorch Tutorial List: [PyTorch Tutorial List](../../programming/pytorch/)
