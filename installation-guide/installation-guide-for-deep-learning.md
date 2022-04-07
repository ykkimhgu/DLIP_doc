---
description: Installation Guide for Deep Learning
---

# Installation Guide

## Installation Guide

### Deep Learning Installation Guide for Win10 <a href="#f126" id="f126"></a>

You must install in the following order. Make sure you install the correct software version as instructed.

> For DLIP 2022-1 Lecture:
>
> * Python 3.7, CUDA 10.2, cuDNN 8.0.5
> * PyTorch 1.10.x
> * Anaconda for Python 3.7 or Anaconda of Latest Version

***

## Installation Steps

(updated 2022.4)

#### 1. Install Anaconda

**Anaconda** : Python and libraries package installer.

Follow: [How to install Anaconda](anaconda.md#conda-installation)

#### 2.Install GPU library (CUDA, cuDNN)

**Nvidia GPU driver** **and Library** : To operate the GPU.

* **CUDA** — GPU C library. Stands for _Compute Unified Device Architecture._
* **cuDNN** — DL primitives library based on CUDA. Stands for _CUDA Deep Neural Network._

Follow [How to install CUDA and cuDNN](cuda-installation.md#9f39)

#### 3. Install Python

> Python 3.7 (2022-1)

Python is already installed by installing Anaconda. But, we will make a virtual environment for a specific Python version.

xit

```c
conda create -n py37tf23 python=3.7
```

![](<../.gitbook/assets/image (311).png>)

#### 4. Install IDE (Visual Studio Code)

[How to Install VS Code](ide/vscode/#installation)

**IDE (VS CODE)**

#### 5 Install DL Framework

**Framework**

* **TensorFlow** — DL library, developed by Google.
* **Keras** — \*\*\*\* DL wrapper with interchangeable backends. Can be used with TensorFlow, Theano or CNTK.
* **PyTorch** — Dynamic DL library with GPU acceleration.

#### 5-1. Install Framework\_1 (Tensorflow and Keras)

Run 'Anaconda Prompt' > activate virtual environment > install tensorflow-gpu 2.3.0 packages > install keras

```c
>>conda activate py37tf23 
>>conda install tensorflow-gpu=2.3.0
>>conda install keras
```

#### 5-2. Install Framework\_2 (Pytorch)

#### 6. Installing other DL libraries

###
