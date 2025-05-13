---
description: (updated 2025.2).
---

# Introduction

## Image Processing with Deep Learning

Deep learning innovations are driving breakthroughs in the field of computer vision for automotive, robotics, and science.

Thus, the aim of this course is to introduce students of computer vision starting from basics of digital image processing and then turning to modern deep learning models, specifically convolutional neural networks(CNN), for image enhancement and analysis applications on Energy/Environment, Autonomous Vehicle/Robots, and Machine Vision.

Students will learn how to build CNN(Convolutional Neural Network) based object detection model for vehicle, pedestrians and other target objects. Also, they will learn how to apply pre-processing and post-processing for CNN model input/output, such as calibration, filtering, image scaling, image augmentation using fundamental methods of computer vision.

This course is composed of two parts:

(1) Part1 : Digital Image Processing (8 weeks) Basics of digital image processing for pre/post-processing of CNN models will be covered including cameral calibration, spatial filtering, feature recognition, color image processing, image scaling/rotating and more. Students will learn how to apply image processing methods to build their own defect inspection and road lane detection system.

(2) Part 2: Convolutional Neural network for Object Detection (8 weeks) Basics and useful technique in building a CNN model will be covered including backpropagation, evaluation metric, regularization and more. Also, the state-of-the-art CNN models such as Inception and Yolo v3 will be introduced. Students will learn how to use their own customized dataset to train and test a CNN object detection model.

After taking this course, students will be able to build their own program for

* Road lane detection
* Vehicle/Pedestrian detection
* Face recognition
* Defect Inspection
* and more

### **Lecture Syllabus**

### **DLIP- 2025**



<table><thead><tr><th width="113">일자</th><th width="64">주차</th><th width="64">Day</th><th width="269">Lecture</th><th width="317">Tutorial</th><th width="233">LAB</th><th width="359">과제 공지</th></tr></thead><tbody><tr><td>03월 04일</td><td>1</td><td>T</td><td>Course Overview<br>Introduction to Image Processing</td><td>TU: OpenCV Basics</td><td>　</td><td>Tutorial: Installation for OpenCV C++</td></tr><tr><td>03월 07일</td><td>　</td><td>F</td><td>　</td><td>TU: OpenCV Basics</td><td>　</td><td>　</td></tr><tr><td>03월 11일</td><td>2</td><td>T</td><td>Camera Optics/<br>Spatial Filter</td><td>　</td><td>　</td><td>TU: Filter (1 week)</td></tr><tr><td>03월 14일</td><td>　</td><td>F</td><td>　</td><td>TU: Filter</td><td>　</td><td>　</td></tr><tr><td>03월 18일</td><td>3</td><td>T</td><td>Histogram/<br>Threshold &#x26; Morphology</td><td>TU: Thresholding_Morphology </td><td>　</td><td>　</td></tr><tr><td>03월 21일</td><td>　</td><td>F</td><td>　</td><td>　</td><td>Quiz1 - written<br>LAB: GrayScale Image Segmentation</td><td>LAB1: Grayscale image (2week)</td></tr><tr><td>03월 25일</td><td>4</td><td>T</td><td>　</td><td>　</td><td>LAB: GrayScale Image Segmentation</td><td>　</td></tr><tr><td>03월 28일</td><td>　</td><td>F</td><td>Edge,Line,Corner Detection</td><td>　</td><td>　</td><td>TU: Line, Edge detection(1 week)</td></tr><tr><td>04월 01일</td><td>5</td><td>T</td><td>TU: Line, Edge detection</td><td>　</td><td>　</td><td>　</td></tr><tr><td>04월 04일</td><td>　</td><td>F</td><td>Camera Modeling and Calibration</td><td>TU: Calibration</td><td>　</td><td>　</td></tr><tr><td>04월 08일</td><td>6</td><td>T</td><td>Color Image Processing</td><td>TU: Color Image Processing</td><td>LAB: Color Image Processing</td><td>LAB2: Color Image (2week)</td></tr><tr><td>04월 11일</td><td>　</td><td>F</td><td>　</td><td>　</td><td>LAB: Color Image Processing</td><td>　</td></tr><tr><td>04월 15일</td><td>7</td><td>T</td><td>Quiz2 - written</td><td>　</td><td>Test1-Programming</td><td>Installation Guide for Deep Learning 2024</td></tr><tr><td>04월 18일</td><td>　</td><td>F</td><td>　</td><td>TU: OpenCV-Python</td><td>LAB:  Image Processing in Python</td><td>　</td></tr><tr><td>04월 22일</td><td>8</td><td>T</td><td>　</td><td>　</td><td>LAB:  Image Processing in Python</td><td>LAB3: Image Processing in Python  (2 weeks)</td></tr><tr><td>04월 25일</td><td>　</td><td>F</td><td>MLP Introduction</td><td>　</td><td>　</td><td>　</td></tr><tr><td>04월 29일</td><td>9</td><td>T</td><td>Optimization, Loss Function<br>BackPropagation </td><td>　</td><td>　</td><td>　</td></tr><tr><td>05월 02일</td><td>　</td><td>F</td><td>　</td><td>TU: Pytorch Exercise (MLP)</td><td>　</td><td>　</td></tr><tr><td>05월 06일</td><td>10</td><td>T</td><td>(휴일)</td><td>　</td><td>　</td><td>　</td></tr><tr><td>05월 09일</td><td>　</td><td>F</td><td>Overview of CNN,  Most commonly used CNN</td><td>　</td><td>　</td><td>　</td></tr><tr><td>05월 13일</td><td>11</td><td>T</td><td>CNN (loss function &#x26; regualization, evaluation)</td><td>TU: Pytorch (LeNet-5) // T2-1<br>TU: Pytorch (VGG) // T2-2 </td><td>　</td><td>　</td></tr><tr><td>05월 16일</td><td>　</td><td>F</td><td>(Special Lecture)</td><td>TU: Pytorch(Pretrained) T3-1, T3-2<br>Assignment (T3-3, T3-4) Classification </td><td>　</td><td>　</td></tr><tr><td>05월 20일</td><td>12</td><td>T</td><td>Object Detection CNN models</td><td>TU: YOLOv8 in Pytorch (Install/Train/Test)  T4-1,4-2</td><td>　</td><td>LAB4: Object Detection(2weeks)</td></tr><tr><td>05월 23일</td><td>　</td><td>F</td><td>　</td><td>　</td><td>Quiz 3- Written  &#x26;  Test 2:  Programming</td><td>　</td></tr><tr><td>05월 27일</td><td>13</td><td>T</td><td>Special Topics in Deep Learning</td><td>(Presentation) Final Lab Proposal</td><td>Final Lab</td><td>LAB5: Final Lab ( Report by 16th week)</td></tr><tr><td>05월 30일</td><td>　</td><td>F</td><td>　</td><td>　</td><td>Final Lab</td><td>　</td></tr><tr><td>06월 03일</td><td>14</td><td>T</td><td>　</td><td>　</td><td>Progressive Presentation</td><td>　</td></tr><tr><td>06월 06일</td><td>　</td><td>F</td><td>(휴일)</td><td>　</td><td>Final Lab</td><td>　</td></tr><tr><td>06월 10일</td><td>15</td><td>T</td><td>　</td><td>　</td><td>Final Lab</td><td>　</td></tr><tr><td>06월 13일</td><td>　</td><td>F</td><td>　</td><td>(Presentation) Final Presentation</td><td>Demonstration</td><td>　</td></tr><tr><td>06월 17일</td><td>16</td><td>T</td><td>　</td><td>　</td><td>Report Due</td><td>　</td></tr><tr><td>06월 20일</td><td>　</td><td>F</td><td>　</td><td>　</td><td>　</td><td>　</td></tr></tbody></table>



#### **Part 1**

* Introduction to Image Processing
* Optics / Calibration
* Tutorial: OpenCV, Calibration
* Spatial Filtering
* Histogram/Threshold
* Morphology
* Edge,Line,Corner Detection
* Color Image Processing

**Part 2**

* Inro. Deep Neural Network: MLP, Activation Function, Loss Function
* Inro. Deep Neural Network: Back Propagation, CaseStudy(LeNet)
* Intro. Convolutional Neural Network: Convolution, Evaluation Metric
* Intro. Convolutional Neural Network:Strategy for CNN design, Case Study(AlexNet)
* Popular CNN models: VGGNet, ResNet, Inception
* Object Detection: from R-CNN to YOLO
* Recent trend in CNN

#### **Tutorial**

* Tutorial: OpenCV, Calibration
* Tutorial: Image Spatial Filtering
* Tutorial: Morphology
* Tutorial: Edge & Line Detection
* Tutorial: Color Image Processing
* Tutorial: PyTorch
* Tutorial: LeNet
* Tutorial: AlexNet
* Tutorial: AlexNet with Customized Dataset
* Tutorial: YOLOv8

### **LAB**

* LAB: Object Segmentation in Grayscale Image
* LAB: Object Segmentation in Color Image
* LAB: Industrial Problem
* LAB: CNN Object Detection (Vehicle /Pedestrian)
* LAB: CNN Object Detection (Custom dataset, Free Topic)
