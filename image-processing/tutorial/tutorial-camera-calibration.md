# Tutorial: Camera Calibration

[calibration resources](https://github.com/ykkimhgu/DLIP_doc/files/14879534/calibration.resources.zip)

DLIP Tutorial for Camera Calibration using MATLAB

**Using **_**"Computer Vision Toolbox"**_** Application in MATLAB**


## Tutorial - Calibration using MATLAB Toolbox

1. Download _**Computer Vision Toolbox**_ in MATLAB.

![img3](https://user-images.githubusercontent.com/84509483/226327538-cb410359-6337-4030-b6fd-83042b1db028.PNG)



2. Open the _**Camera Calibrator**_ application.

![img2](https://user-images.githubusercontent.com/84509483/226327602-6d01d8c2-bf21-4fb0-812c-c6438fec07ba.PNG)



3. Download images for camera calibration. [(link)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Calibration/camera\_calibration\_images.zip)



4. Load calibration images to the camera calibrator app.

![img](https://user-images.githubusercontent.com/84509483/226327653-216ad6ed-34ea-4fab-bd60-98499c6e18c7.PNG)



5. Configure Image and Pattern Properties as

* Pattern Selection:  Checkerboard
* Size of checkerboard square:  25 mm
* Image distortion: Low

![img4](https://user-images.githubusercontent.com/84509483/226327686-7ee6cf2d-e079-4b28-9e30-1db0482f04a9.PNG)



6. Click _**Calibrate**_ button.

![img5](https://user-images.githubusercontent.com/84509483/226327718-35316e83-78bc-4d68-aa43-ee61d96d16ac.PNG)



7. Export Parameters to workspace
8. Save the Workspace  _**cameraParams**_  as **"cameraParams.mat"**

![img6](https://user-images.githubusercontent.com/84509483/226327732-b066f4a1-fc5e-4d07-8d66-f5ddfafb2acb.PNG)



8. Download test code([link](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Calibration/DLIP\_Tutorial\_Calibration\_GetUndistortedImg.m)) and Run the code

![img7](https://user-images.githubusercontent.com/84509483/226327756-702956a0-f1d7-4098-a7fb-2b149f31df37.PNG)



9. Apply the camera parameter values from _**cameraParams**_ to the cpp test code ([link](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Calibration/ShowUndistorted.cpp))

![img8](https://user-images.githubusercontent.com/84509483/226327795-2cf5e1fc-e856-4a53-8c23-625d71ad43ff.PNG)

10. (Option) Create a simple function that returns undistort output image from the input raw image



### Other Calibration Tutorial

1. Calibration with OpenCV C++

{% embed url="https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html" %}

2. Calibration with OpenCV-Python

{% embed url="https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html" %}
