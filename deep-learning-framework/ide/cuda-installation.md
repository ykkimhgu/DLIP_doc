# CUDA Installation

## Installing Tensorflow with CUDA, cuDNN and GPU support on Windows 10 <a id="9f39"></a>

This covers the installation of CUDA, cuDNN on Windows 10. This article below assumes that you have a CUDA-compatible GPU already installed on your PC;

{% embed url="https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html" %}



## Step 1: Check the software you will need to install <a id="0330"></a>

Assuming that Windows is already installed on your PC, the additional bits of software you will install as part of these steps are:-

* Microsoft Visual Studio \(v2017 or 2019\)
* the NVIDIA CUDA Toolkit
* NVIDIA cuDNN

## Step 2: Download Visual Studio Express <a id="d390"></a>

### Visual Studio is a Prerequisite for CUDA Toolkit <a id="bf6e"></a>

Visual studio is required for the installation of Nvidia CUDA Toolkit \(this prerequisite is referred to [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)\). If you attempt to download and install CUDA Toolkit for Windows without having first installed Visual Studio, you get the message shown in Fig. 1.

See how to install Visual Studio 2017/2019



## Step 3: Download CUDA Toolkit for Windows 10 <a id="2582"></a>

> \(updated 2021.4 : Install CUDA Toolkit 10.2 \)

These CUDA installation steps are loosely based on the[ Nvidia CUDA installation guide for windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). The CUDA Toolkit \(free\) can be downloaded from the Nvidia website**.**

**For CUDA Toolkit 10.2**

{% embed url="https://developer.nvidia.com/cuda-10.2-download-archive" %}

> You should check which version of CUDA Toolkit you choose for download and installation to ensure compatibility with Pytorch or [Tensorflow](https://www.tensorflow.org/install/gpu)
>
> **For Latest CUDA Toolkit :** [**click here**](https://developer.nvidia.com/cuda-downloads)\*\*\*\*

Select  Window10,  exe\(Network\). Then, download the installation "exe"file.

![](../../.gitbook/assets/image%20%28106%29.png)

![](../../.gitbook/assets/image%20%28111%29.png)

After Downloading, Install\(Recommended Option\). It should take 10~30min to install.

![](../../.gitbook/assets/image%20%28117%29.png)

