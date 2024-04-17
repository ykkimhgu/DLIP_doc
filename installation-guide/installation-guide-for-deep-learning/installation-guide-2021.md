# Installation Guide 2021

## Installation Guide 2021

***

### description: Installation Guide for Deep Learning 2021

## Installation Guide

### Installation Guide for Win10

This installation guide is for programming a deep learning application using Pytorch or Tensorflow.

Make sure you install the correct software version as instructed.

> For DLIP 2021-1 Lecture:
>
> * Python 3.7, CUDA 10.2, cuDNN 8.0.5
> * PyTorch 1.10.x
> * Anaconda for Python 3.7 or Anaconda of Latest Version

***

### Installation Steps

(updated 2021.4)

#### 1. Install Anaconda

**Anaconda** : Python and libraries package installer.

Follow: [How to install Anaconda](../anaconda.md#conda-installation)

####

#### 2. Install Python & Numpy & OpenCV

**Install Python**

> Python 3.7 (2022-1)

Python is already installed by installing Anaconda. But, we will make a virtual environment for a specific Python version.

* Open Anaconda Prompt(admin mode)
* First, update conda

![](https://user-images.githubusercontent.com/38373000/162147626-98c7c618-2882-4668-a61d-0682cffdd898.png)

```c
conda update -n base -c defaults conda
```

* Then, Create Virtual environment for Python 3.7. Name the $ENV as `py37`

```c
conda create -n py37 python=3.7
```

![](https://user-images.githubusercontent.com/38373000/162149298-8e254ebd-c698-4ab9-bb80-40b24ce2b438.png)

After installation, activate the newly created environment

```c
conda activate py37
```

![](https://user-images.githubusercontent.com/38373000/162150172-0192d3d4-901f-4356-8c99-ff146297bd39.png)

**Install Numpy, OpenCV, Matplot**

```
conda activate py37

conda install numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge opencv
```

> If installed Numpy is not recognized after installation with `conda`, then install Numpy using `pip`
>
> `pip install numpy`

####

#### 3. Install IDE (Visual Studio Code)

Follow: [How to Install VS Code](../ide/vscode/#installation)

Also, read about

* [How to program Python in VS Code](https://ykkim.gitbook.io/dlip/installation-guide/ide/vscode/python-vscode)
* [How to program CoLab(Notebook) in VS Code](https://ykkim.gitbook.io/dlip/installation-guide/ide/vscode/notebook-with-vscode)

####

#### 4. Install GPU library (CUDA, cuDNN)

Skip this if you do not have GPU.

**Nvidia GPU driver** **and Library** : To operate the GPU.

* **CUDA** — GPU C library. Stands for _Compute Unified Device Architecture._
* **cuDNN** — DL primitives library based on CUDA. Stands for _CUDA Deep Neural Network._

Follow [How to install CUDA and cuDNN](../cuda-installation/#9f39)

####

#### 5. Install DL Framework

**Framework**

* **TensorFlow** — DL library, developed by Google.
* **Keras** — DL wrapper with interchangeable backends. Can be used with TensorFlow, Theano or CNTK.
* **PyTorch** — Dynamic DL library with GPU acceleration.

**Install Pytorch**

Read more [about PyTorch installation](https://ykkim.gitbook.io/dlip/installation-guide/framework/pytorch)

* **With GPU**

```
# CUDA 10.2
conda activate py37
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

> Latest PyTorch does not support CUDA 10.2 . please use CUDA-11.3 for Latest version.

* **Without GPU**

```
# CPU Only
conda activate py37
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch
```

**Install Tensorflow and Keras**

* Run 'Anaconda Prompt(admin)'
* Activate virtual environment
* install tensorflow-gpu 2.3.0 packages
* install keras

```c
conda create -n py37tf23 python=3.7
conda activate py37tf23 
conda install tensorflow-gpu=2.3.0
conda install keras
```

####

#### 6. Installing Other libraries

```
conda activate py37

conda install -c conda-forge matplotlib
conda install -c conda-forge opencv
conda install -c anaconda scikit-learn
conda install -c anaconda pandas
conda install jupyter
conda install -c anaconda ipykernel
```
