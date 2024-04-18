# CUDA cuDNN

## 

# Installation of  CUDA, cuDNN 

(updated 2024.4)

**For DLIP 2024 course,** 

>  CUDA=11.8,  PyTorch 2.0





This covers the installation of CUDA, cuDNN on Windows 10/11. 

Here, we assume that you have a CUDA-compatible GPU already installed on your PC.

[See here for more detailed instruction by NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)



The order of installation is 

1. NVIDIA Graphic Card Driver
2. CUDA toolkit and cuDNN

***





# 1. Installation of NVIDIA Graphic Card Driver



### Prerequisite:  Visual Studio Community

**Visual Studio is a Prerequisite for CUDA Toolkit**&#x20;

Visual Studio Community is required for the installation of Nvidia CUDA Toolkit. If you attempt to download and install CUDA Toolkit for Windows without having first installed Visual Studio, you will get a message for the installation.

Follow: [How to install Visual Studio Community](../ide/visual-studio-community.md#how-to-install)





### Step 1: Find Your GPU Model 

 You can check your GPU Card in Window Device Manager



|   항목    |                     Graphics Card  Info                      |                        Window Version                        |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Short Key |            Win Key →  Device Manager (장치관리자)            |          Win key→ System  (또는 내PC 우클릭 → 속성)          |
|           | ![image](https://user-images.githubusercontent.com/23421059/169219424-f8238a68-5129-4c03-a2fd-2538348c8079.png) | ![image](https://user-images.githubusercontent.com/23421059/169219451-b6e6f76a-0e38-4207-8ad9-5963c0dc1def.png) |





### Step 2: Check If Graphic Driver is Installed

Go to  `anaconda prompt` Admin Mode.

> If Anaconda is not installed, see here [Anaconda Installation](https://ykkim.gitbook.io/dlip/installation-guide/anaconda#conda-installation)



You can also use r Window Command Prompt 

> **Start > All Programs > Accessories > Command Prompt**



```
nvidia-smi
```

![image](https://user-images.githubusercontent.com/23421059/169212558-43a032d0-e1c1-4a35-94cf-564701525668.png)



If you have the summary of CUDA version, you already have installed the driver. 

Here, CUDA version 11.6 means, the driver can support up to CUDA ~11.6.



If you don't see any information, go to Step 3. Otherwise Skip Step 3





### Step 3:  Download NVIDIA Graphic Driver 

Download Site: https://www.nvidia.co.kr/Download/index.aspx?lang=kr



Select your Graphic Card Model and Window Version.  Then, download.



![image](https://user-images.githubusercontent.com/23421059/169218227-26c040fd-1c7e-457d-921e-fcd535b4816b.png)



It does not matter whether you select  (a) GRD(game-ready driver)   (b) NVIDIA Studio Driver. 



![image](https://user-images.githubusercontent.com/23421059/169220103-82df5ba9-dc0b-4e94-a0b1-28132c2713c3.png)



Now, Install.    There is NO need to select GeForce Experience.  Choose the default Settings.



![image](https://user-images.githubusercontent.com/23421059/169220499-a244b3ca-e676-4096-a98b-0732259db7a9.png)





Check if you have installed, check the driver version. 

Go to  `anaconda prompt` Admin Mode. Then, go to the environment you want to work with. 



```
conda activate py39
nvidia-smi
```

<img width="656" alt="image" src="https://github.com/ykkimhgu/DLIP_doc/assets/38373000/09abba07-1a38-41ab-ab4e-9733d3bca77d">





# 2. Install CUDA & CuDNN using Conda

## Step 1.  Find supported  CUDA version

Depending on your graphic card, there is a minimum version for CUDA SDK support.



> 

First, find the range of CUDA SDK versions for your Graphic Card (Compatible version)

Example: 

* GTX 1080  is PASCAL(v6.1) and it  is supported by  CUDA SDK from 8.0 to 12.4 

* RTX 40xx is AdaLovelace(v8.9) and it is supported by  CUDA SDK from  11.8 to 12.4 

  

> For most GPUs, CUDA SDK >11.x will work fine
>
> BUT, if you use RTX4xxs, you may need to install CUDA SDK > 11.8



See here for detail https://en.wikipedia.org/wiki/CUDA

![image](https://github.com/ykkimhgu/DLIP_doc/assets/38373000/0f0e8418-efb2-45cc-aef3-2998c0dc11d9)



## Step 2. Install CUDA and cuDNN via CONDA

It is recommended to install **specific** CUDA version in the **selected Python environment.**

> CUDA=11.8,  for DLIP 2024-1 
>
> CUDA=11.2, for DLIP 2023-1 
>
> CUDA=10.2.89, for DLIP 2023-1 



Run Anaconda Prompt(admistration).

Activate conda virtual environment. Then, Install specific CUDA version

> \[$ENV\_NAME] is your environment name. e.g. `conda activate py39`



### DLIP 2024-1

```c
#conda activate [$ENV_NAME]
conda activate py39
    
# CUDA 11.8 & CuDNN
conda install -c anaconda cudatoolkit==10.2.89 cudnn 
```



### DLIP 2022-1

```c
#conda activate [$ENV_NAME]
conda activate py39
    
# CUDA 10.2 & CuDNN
conda install -c anaconda cudatoolkit==10.2.89 cudnn 
```





### Important Note

Depending on your CUDA version, the minimum version of PyTorch is determined. 

For example

*  CUDA 11.6 supports PyTorch 1.13 or higher

* CUDA 11.8 supports PyTorch 2.0 or higher 

  

See here for CUDA and PyTorch Version matching

https://pytorch.org/get-started/previous-versions/



**For DLIP 2024 course,** 

>  CUDA=11.8,  PyTorch 2.0

