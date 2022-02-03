# MATLAB-OpenCV

## Using OpenCV in Matlab

**Computer Vision Toolbox OpenCV Interface**

{% hint style="info" %}
**Requirement: Computer Vision Toolbox**
{% endhint %}

The Computer Vision System Toolbox OpenCV Interface enables you to bring existing OpenCV files and projects into MATLAB using MEX. The support package includes:\
• Data type conversions between MATLAB and OpenCV\
• Examples showing how to interface OpenCV and MATLAB\
Opening the .mlpkginstall file from your operating system or from within MATLAB will initiate the installation process available for the release you have. This .mlpkginstall file is functional for R2014b and beyond.\
Watch this video to learn more about the package: [http://youtu.be/BasC2jkgyaM](http://youtu.be/BasC2jkgyaM)

{% embed url="http://youtu.be/BasC2jkgyaM" %}

**How to Install**

* Type in the following in Matlab Command Line

```
>> visionSupportPackages
```

* Select  **Computer Vision Toolbox OpenCV Interface** and install
* After installation setup the compiler by `mex -setup c++`
* Choose the visual studio C++ compiler version (e.g. V.S 2017)
* Check where the toolbox package  is located in your computer  `which mexOpenCV`

**How to Use**

* You can get started using this quick command-line example: (MATLAB 2020a)

```
%To run the Oriented FAST and Rotated BRIEF (ORB) example, follow these steps:

%1. Change your current working folder to example/ORB where source files
%detectORBFeaturesOCV.cpp and extractORBFeaturesOCV.cpp are located

%2. Create MEX-file for the detector from the source file:
mexOpenCV detectORBFeaturesOCV.cpp

%3. Create MEX-file for the extractor from source file:
mexOpenCV extractORBFeaturesOCV.cpp

%4. Run the test script:
testORBFeaturesOCV.m 

%The test script uses the generated MEX-files.
```

* Refer to  **MATLAB\_OpenCV examples**

{% embed url="https://www.mathworks.com/help/vision/ug/opencv-interface_bujcprv.html#bujcpsp" %}

* Read  **OpenCV with MATLAB** for more info

{% embed url="https://www.mathworks.com/discovery/matlab-opencv.html" %}
