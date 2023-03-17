# CUDA 10.2

## CUDA cuDNN Installation

### Installing CUDA, cuDNN on Windows 10

This covers the installation of CUDA, cuDNN on Windows 10. This article below assumes that you have a CUDA-compatible GPU already installed on your PC.

[See here for more detailed instruction by NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

## Method 1: Installation using Anaconda

Install CUDA and cuDNN with `conda` in Anaconda prompt.

Here, it is assumed you have already installed Anaconda. If you do not have Anaconda installed, follow [How to Install Anaconda](https://ykkim.gitbook.io/dlip/dlip-installation-guide/cuda-installation)

> Recommend to use conda virtual environment for specific CUDA version contained in the env.

### Installing CUDA =10.2

First, Run Anaconda Prompt(admistration)

**(Option1: install in base)**

If you want to install the same CUDA version for all environment, install in (base)

```c
conda install -c anaconda cudatoolkit==10.2.89
```

**(Option2: install in Specific Environment)**

It is recommended to install specific CUDA version in the selected Python environment.

> \[$ENV\_NAME] is your environment name. e.g. `conda activate py37`

```c
#conda activate [$ENV_NAME]
conda activate py37
conda install -c anaconda cudatoolkit==10.2.89
```

###

### Installing cuDNN

Conda will find the compatible cuDNN for the installed version of CUDA

```c
conda install -c anaconda cudnn
```

The available cuDNN version could be a low version (e.g. cuDNN 7.6.5). If you want to install a higher version, then follow Method 2: Using NVDIA downloader.

![image](https://user-images.githubusercontent.com/38373000/162138066-87f63943-66f7-49b3-836e-f7423bba69e2.png)

##

## Method 2: Installation using NVIDIA downloader

### Step 1: Check the software you need to install <a href="#0330" id="0330"></a>

Assuming that Windows is already installed on your PC, the additional bits of software you will install as part of these steps are:

* Microsoft Visual Studio Community(v2017 or higher )
* NVIDIA CUDA Toolkit
* NVIDIA cuDNN

### &#x20;<a href="#d390" id="d390"></a>

### Step 2: Install Visual Studio Community <a href="#d390" id="d390"></a>

#### Visual Studio is a Prerequisite for CUDA Toolkit <a href="#bf6e" id="bf6e"></a>

Visual Studio Community is required for the installation of Nvidia CUDA Toolkit. If you attempt to download and install CUDA Toolkit for Windows without having first installed Visual Studio, you get a message for installation.

Follow: [How to install Visual Studio Community](../ide/visual-studio-community.md#how-to-install)

### &#x20;<a href="#2582" id="2582"></a>

### Step 3: Install CUDA Toolkit for Windows 10 <a href="#2582" id="2582"></a>

> (updated 2022.2 : Install CUDA Toolkit 10.2 )

For more detailed instructions, see[ Nvidia CUDA installation guide for windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

**Installing CUDA Toolkit 10.2**

* Go to CUDA download center
  * The CUDA Toolkit (free) can be downloaded from the Nvidia website. Click here to go to the download center

{% embed url="https://developer.nvidia.com/cuda-10.2-download-archive" %}

> You should check which version of CUDA Toolkit you choose for download and installation to ensure compatibility with Pytorch or [Tensorflow](https://www.tensorflow.org/install/gpu)
>
> **For Latest CUDA Toolkit :** [**check here**](https://developer.nvidia.com/cuda-downloads)

* Select **Window10 ,** **exe(Network)**. Download the Base Installer and Run.

![](<../../.gitbook/assets/image (106).png>)

![](<../../.gitbook/assets/image (111).png>)

* After Downloading, Install(Recommended Option). It should take 10\~20min to install.
  * It will be installed in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2`

![](<../../.gitbook/assets/image (139).png>)

![](<../../.gitbook/assets/image (117).png>)

### Step 4: Install CUDA patches <a href="#3873" id="3873"></a>

After CUDA installation, install additional Patches for CUDA Toolkit.

![](<../../.gitbook/assets/image (125).png>)

### Step 5: Install cuDNN <a href="#3fc4" id="3fc4"></a>

After installing CUDA and its patches, the next step is to find a compatible version of CuDNN.

* Check which version of \*\*\*\* cuDNN is needed for specific Tensorflow or Pytorch

***

**Step 5.1: Register NVIDIA**

Visit [CuDNN website](https://developer.nvidia.com/cudnn) to download.

First, you have to register to become a member of the NVIDIA Developer Program (free).

![](https://miro.medium.com/max/1803/1\*cXR4ODZGhaoR1rXRmvbU6A.png)

**Step 5.2: Install cuDNN 8.0.5 for CUDA 10.2**

Select Download cuDNN.

![image](https://user-images.githubusercontent.com/38373000/162129708-5dbc70fe-f74c-45c8-af3a-2cf8b6fec75e.png)

Select cuDNN version for CUDA 10.2. You can also check cuDNN Archive if you cannot find the version.

> cuDNN v8.0.5 for CUDA 10.2 or
>
> cuDNN v8.3.3 for CUDA 10.2

![image](https://user-images.githubusercontent.com/38373000/162131292-cfe61536-a14a-43fd-8a8d-aa728ae79533.png)

1.  Unzip the cuDNN package.

    ```
    cudnn-windows-x86_64-*-archive.zip
    ```
2.  Copy the following files from the unzipped package into the NVIDIA CUDA directory.

    * CUDA directory: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2

    Do not replace the whole folders. Just copy the files.

![](https://user-images.githubusercontent.com/38373000/162139770-10184974-4eb4-408c-8ef6-e34a550a918b.png)

3\. Check the installed CUDA in your computer. Run the Command Prompt and type

```
nvcc --version
```

![](<../../.gitbook/assets/image (148) (1).png>)
