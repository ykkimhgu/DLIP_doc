# LAB: CNN Object Detection 1

## Parking Management System

Vehicle counting using a CNN based object detection model

{% embed url="https://www.youtube.com/watch?v=0_IQlXTnO7k" %}

## I. Introduction

In this lab, you are required to create a simple program that (1) counts the number of vehicles in the parking lot and (2) display the number of available parking space.

For the given dataset, the maximum available parking space is 13. If the current number of vehicles is more than 13, then, the available space should display as ‘0’.

![image](https://user-images.githubusercontent.com/38373000/168618818-54cae273-6bb4-40b6-99c8-938e5b5ab54e.png)

### Guidelines

The whole code should be programmed using OpenCV-Python and Pytorch.

* DO NOT copy a project from online sites.
* You can refer to any online material and github repository for assistance and getting ideas with proper reference citation.
* Use pretrained YOLO v8 or lastest version.
  * You can also use any other pretrained object detection models
  * You can also train a model using custom datasets
*

**Warning!**

Your lab will not be scored if

```
* your program does not run 
```

* If copied from the lab of previous years or from your classmates
* or any other plagiarism

​

## II. Procedure

### Dataset Preparation

* Download the test video file: [click here to download](https://drive.google.com/file/d/1d5RATQdvzRneSxvT1plXxgZI13-334Lt/view?usp=sharing)
* Download the label(answer) file: [download file](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB-ParkingSpace/LAB_Parking_counting_result_answer_student_modified.txt)

### Instruction

* Need to count the number of vehicles in the parking lot for each frame
  * DO NOT COUNT the vehicles outside the parking spaces
  * Consider the vehicle is outside the parking area if the car's center is outside the parking space
* Make sure you do not count duplicates of the same vehicle
* It should accurately display the current number of vehicle and available parking spaces
*   Save the vehicle counting results in '**counting\_result.txt'** file.

    * When your program is executed, the 'counting\_result.txt' file should be created automatically for a given input video.
    * Each line in text file('counting\_result.txt') should be the pair of **frame# and number of detected car**.
    * Frame number should start from 0.

    ex) 0, 12 1, 12 ...
* In the report, you must evaluate the model performance with numbers (accuracy etc)
  * See the label(answer) file for Frame 0 to Frame 1500&#x20;
* Your program will be scored depending on the accuracy of the vehicle numbers
  * TA will check the Frame 0 to the last frame

## III. Report and Demo Video

This lab will be scored depending on the Contents, Complexity, and Completeness .

You are required to write a concise report and submit the program files and the demo video.

### Report

The lab report must be written as a 'Tutorial' format to explain the whole process A to Z in detail.

* Use the report template given here: [https://ykkim.gitbook.io/dlip/dlip-project/report-template](https://ykkim.gitbook.io/dlip/dlip-project/report-template)
* Also, refer to another examples of tutorials: [example 1](https://keras.io/examples/vision/retinanet/), [example 2](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb)
* Write the report in markdown ‘\*.md’ format
  * You can also write in ''\*.ipynb' format
* You need to include concise explanations and codes for each process in the report
* You should embed code snippets where necessary
* You can also embed your demo video in the report

### Demo Video

You must create a demo video that shows the bounding box of the cars within the parking space only.

Write down the youtube link in the report.&#x20;

### Submission Check List

1. Zip file of report and codes

* Zip file named as : `DLIP_LAB_PARKING_21700000_홍길동.zip`&#x20;
* The Zip file includes
  * Report (\*.pdf)
  * src codes (all source codes should be under `/src` folder)
  * counting\_result.txt

2. Demo Video

* Link on the report.&#x20;
* Don't need to submit the video file.&#x20;

