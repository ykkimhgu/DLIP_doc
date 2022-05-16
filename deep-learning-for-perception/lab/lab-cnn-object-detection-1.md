# LAB: CNN Object Detection 1



# Parking Management System 

Vehicle counting using a CNN based object detection model



# I. Introduction

In this lab, you are required to create a simple program that (1) counts the number of vehicles in the parking lot and (2) display the number of available parking space. 

For the given dataset, the maximum available parking space is 13. If the current number of vehicles is more than 13, then, the available space should display as ‘0’.



## Guidelines

The whole code should be programmed using OpenCV-Python and Pytorch. 

* DO NOT copy a project from online sites.
* You can refer to any online material and github repository for assistance and getting ideas with proper reference citation. 
* Use pretrained YOLO v5.
  * You can also use any pretrained object detection model, such as YOLO v3~4
  * You can also train the model using custom/other open datasets
* You can clone a github repository of the object detection model(e.g. YOLOv5), as long as you cite it in the reference.

**Warning!**

Your lab will not be scored if 

	* your program does not run 
 * If copied from the lab of previous years or from your classmates
 * or any other plagiarism

​       

# II. Procedure

* Download the test video file: [click here to download](https://drive.google.com/file/d/1d5RATQdvzRneSxvT1plXxgZI13-334Lt/view?usp=sharing)

* Need to count the number of vehicles in the parking lot for each frame

  * DO NOT COUNT the vehicles outside the parking spaces
  * Consider the vehicle is outside the parking area if the car's center is outside the parking space

* Make sure you do not count duplicates of the same vehicle

* It should accurately display the current number of vehicle and available parking spaces

* Save the vehicle counting results in '**counting_result.txt'** file.

  * When your program is executed, the 'counting_result.txt' file should be created automatically for a given input video.
  * Each line in text file('counting_result.txt') should be the pair of   **frame# and number of detected car**. 
  * Frame number should start from 0. 

  ex)
  0, 12
  1, 12
  ...


* Your program will be scored depending on the accuracy of the detected car

 



# III. Report and Demo Video

This lab will be scored depending on the Contents, Complexity, and Completeness .

You are required to write a concise report and submit the program files and the demo video.





## Report 

The lab report must be written as a 'Tutorial' format to explain the whole process A to Z in detail. 

* Use the report template given here: https://ykkim.gitbook.io/dlip/dlip-project/report-template  

*  Also, see example tutorials: [example 1](https://keras.io/examples/vision/retinanet/), [example 2](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb)



* Write the report in markdown ‘*.md’ format   

  * You can also write in ''*.ipynb' format

* You need to include concise explanations and codes for each process in the report 

* You should embed code snippets where necessary

* You can also embed your demo video in the report

  


## Demo Video

You must create a demo video that shows the bounding box of the cars within the parking space only.

* You can submit video file to TA's email or send the download link



## Submission Check List

1. Zip file of report and codes

* Zip file named as :   `DLIP_LAB_PARKING_21700000_홍길동_21800000_김한동.zip` 

* The Zip file  includes 

  * Report (*.md) or ( * .ipynb)
  * Report (*.pdf)
  * src  (source codes under `/src` folder)
  * counting_result.txt

  

2. Demo Video 

* Video file named as :   `DLIP_LAB_PARKING_VIDEO_21700000_홍길동_21800000_김한동` 

  

  

