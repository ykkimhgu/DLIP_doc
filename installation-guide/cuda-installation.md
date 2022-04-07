# CUDA cuDNN

## Installing  CUDA, cuDNN on Windows 10 <a id="9f39"></a>

This covers the installation of CUDA, cuDNN on Windows 10. This article below assumes that you have a CUDA-compatible GPU already installed on your PC.

[See here for more detailed instruction by NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) 



# Method1: Installation using Anaconda (simple method)

Install CUDA and cuDNN with `conda`  in Anaconda prompt. 

Assumed you have already installed Anaconda. Read: [How to Install Anaconda](https://ykkim.gitbook.io/dlip/dlip-installation-guide/cuda-installation)

> Recommend to use conda virtual environment  for specific CUDA  version contained in the env.



## Installing CUDA =10.2

Run Anaconda Prompt(admistration). 

**(Option1: install in base)** 

If you want to install the same CUDA version for all environment, install in (base) 

Install CUDA=10.2

```c
conda install -c anaconda cudatoolkit==10.2.89
```



**(Option2: install in Env.)** 

If not, then activate your virtual environment. Then, Install CUDA=10.2

> [$ENV_NAME] is your environment name.  e.g.  `conda activate py37`

```c
conda activate [$ENV_NAME]
conda install -c anaconda cudatoolkit==10.2.89
```




## Installing cuDNN

Conda will find the compatible cuDNN for the installed version of CUDA

The available cuDNN version could be a low version (e.g. cuDNN 7.6.5). 

```c
conda install -c anaconda cudnn
```



If you want to install a higher version, then follow Method 2.

![image](https://user-images.githubusercontent.com/38373000/162138066-87f63943-66f7-49b3-836e-f7423bba69e2.png)





# Method 2: Installation using NVIDIA downloader

## Step 1: Check the software you will need to install <a id="0330"></a>

Assuming that Windows is already installed on your PC, the additional bits of software you will install as part of these steps are:-

* Microsoft Visual Studio Community\(v2017 or higher \)
* NVIDIA CUDA Toolkit
* NVIDIA cuDNN

## Step 2: Install Visual Studio Community <a id="d390"></a>

### Visual Studio is a Prerequisite for CUDA Toolkit <a id="bf6e"></a>

Visual studio Community is required for the installation of Nvidia CUDA Toolkit \(this prerequisite is referred to [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)\). If you attempt to download and install CUDA Toolkit for Windows without having first installed Visual Studio, you get a message for installation.

Follow: [How to install Visual Studio Community](ide/visual-studio-community.md#how-to-install)



## Step 3: Install CUDA Toolkit for Windows 10 <a id="2582"></a>

> \(updated 2021.4 : Install CUDA Toolkit 10.2 \)

The CUDA Toolkit \(free\) can be downloaded from the Nvidia website**.**

* For more detailed instructions, see[ Nvidia CUDA installation guide for windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). 



**For CUDA Toolkit 10.2 Installation**: click here

{% embed url="https://developer.nvidia.com/cuda-10.2-download-archive" %}

> You should check which version of CUDA Toolkit you choose for download and installation to ensure compatibility with Pytorch or [Tensorflow](https://www.tensorflow.org/install/gpu)
>
> **For Latest CUDA Toolkit :** [**check here**](https://developer.nvidia.com/cuda-downloads)



Select  **Window10**,  **exe\(Network\)** . Download the Base Installer and run. 

![](../.gitbook/assets/image%20%28106%29.png)

![](../.gitbook/assets/image%20%28111%29.png)



After Downloading, Install\(Recommended Option\). It should take 10~20min to install.

It will be installed in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2`

![](../.gitbook/assets/image%20%28139%29.png)

![](../.gitbook/assets/image%20%28117%29.png)



## Step 4: Install CUDA patches <a id="3873"></a>

After CUDA installation, install additional Patches for CUDA Toolkit.

![](../.gitbook/assets/image%20%28125%29.png)





## Step 5:  Install cuDNN <a id="3fc4"></a>

After installing CUDA 9.0 base installer and its patches, the next step is to find a compatible version of CuDNN. 

*  cuDNN version at [least 8.0](https://www.tensorflow.org/install/gpu).







 

**Step 5.1: Register NVIDIA**

Visit [CuDNN site](https://developer.nvidia.com/cudnn), to download. 

First, you have to register to become a member of the NVIDIA Developer Program \(free\).

![](https://miro.medium.com/max/1803/1*cXR4ODZGhaoR1rXRmvbU6A.png)

**Step 5.2: Install cuDNN 8.0.5  for CUDA 10.2**

Select Download cuDNN. 

<img width="1040" alt="image" src="https://user-images.githubusercontent.com/38373000/162129708-5dbc70fe-f74c-45c8-af3a-2cf8b6fec75e.png">

Select cuDNN version for CUDA 10.2. You can also check cuDNN Archive if you cannot find the version.

> cuDNN v8.0.5 for CUDA 10.2 or
>
> cuDNN v8.3.3 for CUDA 10.2

  

<img width="1022" alt="image" src="https://user-images.githubusercontent.com/38373000/162131292-cfe61536-a14a-43fd-8a8d-aa728ae79533.png">



1. Unzip the cuDNN package.

   ```
   cudnn-windows-x86_64-*-archive.zip
   ```

2. Copy the following files from the unzipped package into the NVIDIA CUDA directory.

   * CUDA directory:  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2

   Do not replace the whole folders.  Just copy  the files.

   

<img width="587" alt="image" src="https://user-images.githubusercontent.com/38373000/162139770-10184974-4eb4-408c-8ef6-e34a550a918b.png">



Check the installed CUDA in your computer. Run the Command Prompt and type

```C
nvcc --version
```



![image-20220407161144298](C:\Users\ykkim\AppData\Roaming\Typora\typora-user-images\image-20220407161144298.png)