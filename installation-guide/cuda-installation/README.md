# CUDA cuDNN

## CUDA cuDNN

### Installing CUDA, cuDNN on Windows 10 

This covers the installation of CUDA, cuDNN on Windows 10. This article below assumes that you have a CUDA-compatible GPU already installed on your PC.

[See here for more detailed instruction by NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

(updated 2022.5)





# Installation NVIDIA Driver  (필수)



## Prerequisite: Install Visual Studio Community 

**Visual Studio is a Prerequisite for CUDA Toolkit **

Visual Studio Community is required for the installation of Nvidia CUDA Toolkit. If you attempt to download and install CUDA Toolkit for Windows without having first installed Visual Studio, you get a message for installation.

Follow: [How to install Visual Studio Community](../ide/visual-studio-community.md#how-to-install)





## Step 1: Check If Graphic Driver is Installed 

cudatoolkit에서 GPU에 접근하기 위해서는 특정 버전 이상의 그래픽카드 드라이버가 설치되어있어야 합니다.  

### 

먼저 자신의 드라이버 버전 확인을 위해 cmd 창이나 anaconda prompt를 열고 아래를 입력하십시오

```
nvidia-smi
```

![image](https://user-images.githubusercontent.com/23421059/169212558-43a032d0-e1c1-4a35-94cf-564701525668.png)

결과와 같이 자신의 그래픽카드 드라이버 버전을 확인할 수 있습니다. 



만약 `nvidia-smi`에도 아무 결과가 보이지 않으면 드라이버가 미설치된 상태입니다.  

* Go to Step 2



## Step 2: Install Graphic Driver for your PC



1) 본인의 그래픽카드 및 운영체제 정보 확인 

|   항목    |                       그래픽카드 정보                        |                        운영체제 정보                         |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 접근 방법 |                     win키 → 장치 관리자                      |           win키→ 시스템 (또는 내PC  우클릭 → 속성)           |
| 결과확인  | ![image](https://user-images.githubusercontent.com/23421059/169219424-f8238a68-5129-4c03-a2fd-2538348c8079.png) | ![image](https://user-images.githubusercontent.com/23421059/169219451-b6e6f76a-0e38-4207-8ad9-5963c0dc1def.png) |



2) 다운로드 사이트 접속: https://www.nvidia.co.kr/Download/index.aspx?lang=kr

- 확인된 본인의 PC (or 노트북)에 맞는 GPU 제품 및 운영체제를 선택합니다. 다운로드타입은 아무거나 선택하시면 됩니다.

![image](https://user-images.githubusercontent.com/23421059/169218227-26c040fd-1c7e-457d-921e-fcd535b4816b.png)



- 다운로드 타입(GRD or SD) 별로 드라이버를 찾을 수 있으며, 수업진행에는 모두 차질 없으니 검색되는 제품을 다운받으시면 됩니다. 

![image](https://user-images.githubusercontent.com/23421059/169220103-82df5ba9-dc0b-4e94-a0b1-28132c2713c3.png)



- 그래픽 드라이버를 설치합니다. GeForce Experience는 본 수업과는 크게 관련없으니 해제해도 좋습니다.

![image](https://user-images.githubusercontent.com/23421059/169220499-a244b3ca-e676-4096-a98b-0732259db7a9.png)



- 다른 옵션은 초기 설정대로 진행 및 설치를 완료합니다.





## Step 3. 그래픽 드라이버 설치 버전 확인

설치가 완료되면 `anaconda prompt`를 관리자 모드로 열고 아래를 입력하십시오. 설치된 드라이버 버전을 확인할 수 있습니다.

> Anaconda 미설치 경우  [설치방법 참고](https://ykkim.gitbook.io/dlip/installation-guide/anaconda#conda-installation)

```
conda activate py39
nvidia-smi
```

![image](https://user-images.githubusercontent.com/23421059/169212558-43a032d0-e1c1-4a35-94cf-564701525668.png)









# Install CUDA & CuDNN using Conda

Install CUDA and cuDNN with `conda` in Anaconda prompt.

> CUDA=10.2.89,  2022-1 학기 기준

Here, it is assumed you have already installed Anaconda. If you do not have Anaconda installed, follow [How to Install Anaconda](https://ykkim.gitbook.io/dlip/dlip-installation-guide/cuda-installation)



First, Run Anaconda Prompt(admistration)



**Install in Specific Virtual Environment)**

It is recommended to install specific CUDA version in the selected Python environment.&#x20;

> \[$ENV\_NAME] is your environment name. e.g. `conda activate py39`

```c
#conda activate [$ENV_NAME]
conda activate py39
# CUDA 10.2 & CuDNN
conda install -c anaconda cudatoolkit==10.2.89 cudnn 
```



## Next

Install PyTorch and Other library

