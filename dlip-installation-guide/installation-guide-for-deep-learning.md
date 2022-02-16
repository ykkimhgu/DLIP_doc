---
description: Installation Guide for Deep Learning
---

# Installation Guide

## Deep Learning Installation Guide for Win10 <a href="#f126" id="f126"></a>

Install the following software in the order of

![](<../.gitbook/assets/image (146).png>)

1. **Nvidia GPU driver** **and Library** — To operate the GPU.
   * **CUDA** — GPU C library. Stands for _Compute Unified Device Architecture._
   * **cuDNN** — DL primitives library based on CUDA. Stands for _CUDA Deep Neural Network._
2. **Anaconda** — Python and libraries package installer.
3. **Python**
4. **IDE (VS CODE)**
5. **Framework**
   * **TensorFlow** — DL library, developed by Google.
   * **Keras** — \*\*\*\* DL wrapper with interchangeable backends. Can be used with TensorFlow, Theano or CNTK.
   * **PyTorch** — Dynamic DL library with GPU acceleration.
6. **Other libraries**

\---

(updated 2021.4)

## Installation Steps

### **1.** Install GPU library (CUDA, cuDNN)

[How to install CUDA 10.2, cuDNN 8.0.5](cuda-installation.md#9f39)

### **2. Install Anaconda**

[How to install Anaconda](../programming/dl-library-tools/underconstruction-1.md#conda-installation)

### 3. Install Python (3.7)

How to create virtual environment using Conda (you can skip this now)

```c
conda create -n py37tf23 python=3.7
```

![](<../.gitbook/assets/image (311).png>)

### 4. Install IDE (Visual Studio Code)

[How to Install VS Code](ide/vscode/#installation)

### 5-1. Install Framework\_1 (Tensorflow and Keras)

Run 'Anaconda Prompt' > activate virtual environment > install tensorflow-gpu 2.3.0 packages > install keras

```c
>>conda activate py37tf23 
>>conda install tensorflow-gpu=2.3.0
>>conda install keras
```

### 5-2. Install Framework\_2 (Pytorch)

### 6. Installing other DL libraries

##
