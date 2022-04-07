---
description: Installation Guide for Deep Learning
---

# Installation Guide

## Installation Guide for Win10 <a href="#f126" id="f126"></a>

You must install in the following order. Make sure you install the correct software version as instructed.

> For DLIP 2022-1 Lecture:
>
> * Python 3.7, CUDA 10.2, cuDNN 8.0.5
> * PyTorch 1.10.x
> * Anaconda for Python 3.7 or Anaconda of Latest Version

***

# Installation Steps

(updated 2022.4)

## 1. Install Anaconda

**Anaconda** : Python and libraries package installer.

Follow: [How to install Anaconda](anaconda.md#conda-installation)



## 2. Install Python 

> Python 3.7 (2022-1)

Python is already installed by installing Anaconda. But, we will make a virtual environment for a specific Python version.



* Open  Anaconda Prompt(admin mode)

  ![image](https://user-images.githubusercontent.com/38373000/162147626-98c7c618-2882-4668-a61d-0682cffdd898.png)

  

* First, update conda
```c
conda update -n base -c defaults conda
```

* Then, Create Virtual environment for Python 3.7. Name the $ENV as `py37`
```c
conda create -n py37 python=3.7
```

​    <img src="https://user-images.githubusercontent.com/38373000/162149298-8e254ebd-c698-4ab9-bb80-40b24ce2b438.png" alt="image" style="zoom:60%;" />



After installation, activate the newly created environment

```c
conda activate py37
```



<img src="https://user-images.githubusercontent.com/38373000/162150172-0192d3d4-901f-4356-8c99-ff146297bd39.png" alt="image" style="zoom:80%;" />






## 3. Install IDE (Visual Studio Code)

Follow: [How to Install VS Code](ide/vscode/#installation)



Also, read 

* [How to program Python in VS Code](https://ykkim.gitbook.io/dlip/installation-guide/ide/vscode/python-vscode)
* [How to program CoLab(Notebook) in VS Code](https://ykkim.gitbook.io/dlip/installation-guide/ide/vscode/notebook-with-vscode)





## 4. Install GPU library (CUDA, cuDNN)
Skip this if you do not have GPU.

**Nvidia GPU driver** **and Library** : To operate the GPU.

* **CUDA** — GPU C library. Stands for _Compute Unified Device Architecture._
* **cuDNN** — DL primitives library based on CUDA. Stands for _CUDA Deep Neural Network._



Follow [How to install CUDA and cuDNN](cuda-installation.md#9f39)





## 5 Install DL Framework

**Framework**

* **TensorFlow** — DL library, developed by Google.
* **Keras** — DL wrapper with interchangeable backends. Can be used with TensorFlow, Theano or CNTK.
* **PyTorch** — Dynamic DL library with GPU acceleration.



### Install Pytorch

 **With GPU**

```C
# CUDA 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

> Latest PyTorch does not support CUDA 10.2 .  please use CUDA-11.3 for Latest version.



 **Without GPU**

```C
# CPU Only
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch
```



Read more [about PyTorch installation](https://ykkim.gitbook.io/dlip/installation-guide/framework/pytorch)



###  Install Tensorflow and Keras

* Run 'Anaconda Prompt(admin)'
* Activate virtual environment
* install tensorflow-gpu 2.3.0 packages
* install keras

```c
>>conda create -n py37tf23 python=3.7
>>conda activate py37tf23 
>>conda install tensorflow-gpu=2.3.0
>>conda install keras
```





## 6. Installing other DL libraries


