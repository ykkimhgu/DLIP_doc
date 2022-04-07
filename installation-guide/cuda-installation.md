# CUDA cuDNN

## Installing  CUDA, cuDNN on Windows 10 <a id="9f39"></a>

This covers the installation of CUDA, cuDNN on Windows 10. This article below assumes that you have a CUDA-compatible GPU already installed on your PC.

[See here for more detailed instruction by NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) 

## Step 1: Check the software you will need to install <a id="0330"></a>

Assuming that Windows is already installed on your PC, the additional bits of software you will install as part of these steps are:-

* Microsoft Visual Studio \(v2017 or 2019\)
* the NVIDIA CUDA Toolkit
* NVIDIA cuDNN

## Step 2: Download Visual Studio  <a id="d390"></a>

### Visual Studio is a Prerequisite for CUDA Toolkit <a id="bf6e"></a>

Visual studio is required for the installation of Nvidia CUDA Toolkit \(this prerequisite is referred to [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)\). If you attempt to download and install CUDA Toolkit for Windows without having first installed Visual Studio, you get the message.

[How to install Visual Studio 2017/2019](ide/visual-studio-community.md#how-to-install)



## Step 3: Download CUDA Toolkit for Windows 10 <a id="2582"></a>

> \(updated 2021.4 : Install CUDA Toolkit 10.2 \)

These CUDA installation steps are loosely based on the[ Nvidia CUDA installation guide for windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). The CUDA Toolkit \(free\) can be downloaded from the Nvidia website**.**

**For CUDA Toolkit 10.2**

{% embed url="https://developer.nvidia.com/cuda-10.2-download-archive" %}

> You should check which version of CUDA Toolkit you choose for download and installation to ensure compatibility with Pytorch or [Tensorflow](https://www.tensorflow.org/install/gpu)
>
> **For Latest CUDA Toolkit :** [**click here**](https://developer.nvidia.com/cuda-downloads)\*\*\*\*

Select  Window10,  exe\(Network\). Then, download the installation "exe"file.

![](../.gitbook/assets/image%20%28106%29.png)

![](../.gitbook/assets/image%20%28111%29.png)

After Downloading, Install\(Recommended Option\). It should take 10~20min to install.

It will be installed in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2`

![](../.gitbook/assets/image%20%28139%29.png)

![](../.gitbook/assets/image%20%28117%29.png)

## Step 4: Download Windows 10 CUDA patches <a id="3873"></a>

After CUDA installation, install additional Patches for CUDA Toolkit.

![](../.gitbook/assets/image%20%28125%29.png)





## Step 5: Download and Install cuDNN <a id="3fc4"></a>

Having installed CUDA 9.0 base installer and its four patches, the next step is to find a compatible version of CuDNN. 

 a cuDNN version of at [least 7.2](https://www.tensorflow.org/install/gpu).

**Step 5.1: Downloading cuDNN**

In order to [download CuDNN](https://developer.nvidia.com/cudnn), you have to register to become a member of the NVIDIA Developer Program \(which is free\).

![](https://miro.medium.com/max/1803/1*cXR4ODZGhaoR1rXRmvbU6A.png)

**cuDNN 8.0.5  for CUDA 10.2**

![](../.gitbook/assets/image%20%28129%29.png)



