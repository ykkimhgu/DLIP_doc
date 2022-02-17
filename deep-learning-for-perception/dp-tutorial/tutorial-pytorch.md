# Tutorial: PyTorch



# PyTorch



## Installation

1. Install PyTorch for Cuda 10.2 

   Run the command in Anaconda Prompt(administrator mode) Python 3.7 Environment[See here for more detail](https://ykkim.gitbook.io/dlip/dlip-installation-guide/framework/pytorch)

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```



2. Also, watch [Introduction to PyTorch (20min)](https://youtu.be/IC0_FRiX-sw)

<iframe width="560" height="315" src="https://www.youtube.com/embed/IC0_FRiX-sw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



3. Follow Quick-Start Tutorial: [Pytorch Tutorial](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)

- Finish this tutorial before class



4. Read more about 

- What is Tensor in Pytorch?

  



# Class Tutorial


## Tutorial

### MLP 

- T1-1: [Train a simple MLP (MNIST)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_PyTorch_MNIST_MLP_Part1_Train.ipynb)
- T1-2: [Test with loading trained model](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_PyTorch_MNIST_MLP_Part2_Test.ipynb)



### CNN- Classification

**Create simple CNN model** 

- T2-1: Create LeNeT CNN model and Train with openddataset (CIFAR10, Fashion MNIST)

- T2-2: Test with loading trained model

- T2-3: Create and Train a CNN model(VGG-16) with opendataset


**Load torchvision.model** 

- T3-1: Test using Pretrained Model  (VGG, Inception, ResNet)
- T3-2: Train torchvision model with OpenDataset(CIFAR10)
- T3-3: Prepare  Custom Dataset for training



### CNN- Object Detection

**Test Object Detection model provided in pytorch** 

* T4-1: Test using Pretrained YOLOv5 with COCO dataset

**YOLO v5 in PyTorch**

- T4-2: Test using Pretrained YOLOv5 with COCO dataset
- T4-3: Train YOLOv5 with with Custom Dataset



## LAB
### LAB: Classification

Create a CNN model(VGG-19) and Train with Custom Dataset

참고:  https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_VGG_Keras_2021.ipynb


### LAB: Object Detection 

You should design an object detection problem related to your interested area.  For the dataset, you can search kaggle.com or any other open dataset sites.





## Useful Site

[Pytorch-Tutorial](https://github.com/yunjey/pytorch-tutorial)



