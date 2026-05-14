# Tutorial - PyTorch

DLIP Tutorial for Deep Learning

## Preparation

### Installation&#x20;

Follow the installatin instruction: [See here for more detail](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning#part-3.-installing-dl-framework)

> You should install pytorch in a virtual environment

### Introduction Video on PyTorch

[Introduction to PyTorch (20min)](https://youtu.be/IC0_FRiX-sw)

> You need to know 'What is Tensor in Pytorch'

### **Quick-Start Tutorials**

* [Pytorch Tutorial(ENG)](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
* [Pytorch Tutorial(KOR)](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)

***

## List of Tutorials for DLIP

### 1. MLP

#### Numpy (colab)

* T0: [Numpy for a simple MLP for XOR](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_PythonNumpy/Tutorial_Numpy_MLP_XOR_Student.ipynb)

#### PyTorch (VSC)

* T1: [Train a simple MLP and Test with loading trained model (MNIST) in VS code](https://ykkim.gitbook.io/dlip/deep-learning-for-perception/dp-tutorial/tutorial-pytorch-example-code#example-1.-ann-mlp-model-train-test)

#### PyTorch (colab)

* T1: [Train a simple MLP and Test with loading trained model (MNIST)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T1_MNIST_MLP_2024.ipynb)-colab





### 2. CNN Classification: Model Design

#### **(VSC) Create a simple CNN model**

* T2-1: [Create Train LeNeT CNN model for CIFAR10](https://ykkim.gitbook.io/dlip/deep-learning-for-perception/dp-tutorial/tutorial-pytorch-example-code#example-2.-cnn-model-train-test)
* T2-2: [Create a CNN model(VGG-16) for ImageNet](tutorial-pytorch-example-code/tutorial-pytorch-example-code-2025.md#exercise-define-model-vgg-16)
* T2-3: [Create, Train and Test a CNN model(VGG-16) for CIFAR10](tutorial-pytorch-example-code/#exercise-define-model-vgg-16-1)

#### **(Colab) Create a simple CNN model**  (depricated)

* T2-1: [Create LeNeT CNN model, Train with opendataset, and Test with loading trained model (CIFAR10)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_1_LeNet5_CIFAR10_CNN_2024.ipynb)
* T2-2: [Create a CNN model(VGG-16) for ImageNet](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_2_1_VGG16_CNN_2024.ipynb)
* T2-3: [Create, Train and Test a CNN model(VGG-16) for CIFAR10](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T2_2_2_VGG16_CIFAR10_CNN_2024.ipynb)

#### Assignment: CNN Model Design

* [(Assignment) Create VGG-16 and ResNet-50 model for ImageNet](tutorial-pytorch-example-code/#assignment-1-week)





### 3. CNN Classification : Pretrained Model and Transfer Learning

#### **(VSC) Test with pretrained model and Transfer Learning**

* T3-1: [Test using Pretrained Model (VGG, Inception, ResNet)](tutorial-pytorch-pretrained.md#tutorial-inference-using-pre-trained-model-classification)
* T3-2: [Transfer Learning of Pretrained model](tutorial-pytorch-transfer-learnings.md#tutorial-transfer-learning-using-pre-trained-models-classification)

#### **(Colab) Test with pretrained model and Transfer Learning**  (depricated)

* T3-1: [Test using Pretrained Model (VGG, Inception, ResNet)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T3_1_Inference_using_Pre_trained_Model_\(classification\)_2024.ipynb)
* T3-2: [Transfer Learning of Pretrained model](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T3_2_Transfer_Learning_using_Pre_trained_Models_\(classification\)_2024.ipynb)

#### Assignment: Classification

* [(Assignment) Transfer Learning using Pretrained Mdel](tutorial-pytorch-transfer-learnings.md#assignment)





### 4. CNN Object Detection: YOLO

#### **YOLO v26**

* T4:&#x20;

#### **YOLO v8**

* T4-1: [Install and Inference using YOLOv8](tutorial-yolov8-in-pytorch.md#tutorial-yolo-v8-in-pytorch)
* T4-2: [Train and Test using Custom Dataset](tutorial-yolov8-in-pytorch.md#tutorial-yolo-v8-in-pytorch)

#### **YOLO v5**

* T4-1(option1): [Pretrained YOLOv5 with COCO dataset](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/Tutorial_PyTorch_T4_1_Inference_using_Pretrained_YOLOv5_with_COCO_2022.ipynb) (in CoLab)
* T4-1(option2):[ Pretrained YOLOv5 with COCO dataset](tutorial-yolov5-in-pytorch/) (in VS Code, Local PC)
* T4-2: [Train YOLOv5 with a Custom Dataset](tutorial-yolov5-in-pytorch/tutorial-yolov5-train-with-custum-data.md) (in VS Code, Local PC)

***

### Useful Sites

Pytorch tutorial codes: [Pytorch-Tutorial](https://github.com/yunjey/pytorch-tutorial)

Pytorch Tutorial List: [PyTorch Tutorial List](../../programming/pytorch/)
