---
description: Tutorial Title goes here
---

# Project Title
**Date:**

**Author:**

**Github:** repository link

**Demo Video:** Youtube link

***



## I. Introduction

This lab is about \~

Write a short abstract of the project with necessary diagram.





## II. Problem Statement

### 1. Project Objectives

Explain the objectives of the project



This project aims to develop an AI-powered vision system for a waste recycling system that sort the waste type with a robot arm. Specifically, this project needs to 

* Develop a model that can detect and classify a recyclable waste product
  *  4 classes of PET, PVC, Metal, Glass
* Locate and display the waste with a boundary box
* Sort the waste product with a robot arm
* Display the statistics of each waste product



### 2. Expected Outcomes  

Explain what outcome and evaluation index you are going to achieve.

* A classification deep learning model that can recognize waste type
* A robot arm and conveyor belt system that can sort the waste automatically 
* GUI that display the statistics of the fault rate 



### 3. Evaluation Index

| Evaluation Index                            | Goal    | Description              |
| ------------------------------------------- | ------- | ------------------------ |
| 1. Accuracy of object detection             | >90%    | Test image of 500 frames |
| 2. F1-score of Anomaly Classification       | >00     |                          |
| 3. Estimate accuracy of the detection model | >00     |                          |
| 4. Inference Time (FPS)                     | >10 fps | Tested on GTX1080 TI     |




---



## III. Requirements

Write a list of HW/SW requirements.

### 1. Hardware List

* Jetson Nano
* Webcam

### 2. Software  List 

* CUDA 10.1
* cudatoolkit 10.1
* Python 3.8.5
* Pytorch 1.6.0
* Torchvision==0.7.0
* YOLO v5



### 3. Dataset

A brief description of dataset goes here.

**Dataset link:** download here

* For open dataset, just include the download link.

* For custom dataset, include the download link. Also, need to submit to TA

  

  

  



---

## IV. Installation and Procedure

Explain the whole procedure step by step with proper headings and images.

This section is a tutorial that helps the reader to follow the whole procedure



### 1. Hardware Setup

A simple overview how to install the hardware setup. You may skip this if you do not have any hardware.



### 2. Software Installation

Do need to include installation of  {Python, OpenCV, NumPy, PyTorch} , which were covered in class.

But, you should include CLI  for installing libraries(Python etc), which versions are different from the ones used in class.



### 3. Data Preparation

Explain how to download the datasets,  and how to partition train/test sets 




### 4. Train model

Explain how to train the model. Don't need to show the whole source code. 



### 5. Test model

Explain how to test the model. Don't need to show the whole source code. 





---

## V. Method


### 1. Overview

Explain the overview of your algorithm with  proper diagrams and  flow chart.  



### 2. Preprocessing

Explain how you have done preprocessing on the datasets.

Also include other  image processing you have done



### 3. Deep Learning Model

Briefly explain which the deep learning model you have used, with proper citations



### 4. Postprocessing

**Algorithm #1: 0000**

Briefly explain other algorithms you have created 



**Algorithm #2: 0000**

Briefly explain other algorithms you have created 





### 5. Experiment Method 

Explain how you have tested for evaluation

Also include which evaluation Index were used 






---

## VI. Results and Analysis

Show the final results visually (images, graph, table etc)

Analyze the results in terms of accuracy/precision/recall etc..

Explain whether you have achieved the project objectives



---



## VII. Conclusion

Do not write your personal comments. 

This is to summarize the overall project that include the main objectives, the methods, and the final results.

Also, include what should be the further work to improve the project.





## Reference

Complete list of all references used (github, blog, paper, etc)

***





## Appendix



### 1. Team Contribution

| Name | Job Descriptions |
| ---- | ---------------- |
|      |                  |
|      |                  |



### 2. Debugging



### 3. Others



