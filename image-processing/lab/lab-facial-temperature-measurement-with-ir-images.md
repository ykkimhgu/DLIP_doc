---
description: Detect Face Temperature from IR(Infra-Red) images
---

# LAB: Facial Temperature Measurement with IR images

## I. Introduction

In this lab, you are required to create a simple program that detects the temperature of a person wearing a mask. You will be given a video of IR images of several people measuring their face temperature. Measure the maximum and average temperature of the face (excluding the mask) and show a warning sign if the average temperature is above 38.0 C.

We will not use any deep learning or any other complex algorithms. Just use simple image processing methods such as :&#x20;

   InRange, Morphology, Filtering, findContour

   Refer to \[Tutorial: Color Image Segmentation] for programming tips

Download the source Video file:[ Click here](https://github.com/ykkimhgu/DLIP-src/tree/main/LAB\_color)



{% embed url="https://youtu.be/xvM0-H5nXoM" %}



## II. Procedure

### Part 1. Face Segmentation excluding mask

#### Segmentation using InRange()

Recommendation: use the program code given in  \[Tutorial:color segemtation]

* &#x20;Analyze the color space of the raw image. You can use either RGB or HSV space
* &#x20;Apply necessary pre-processing, such as filtering.
* &#x20;By using InRange(), segment the area of ROI: exposed skin (face and neck) that are not covered by cloth and mask.  You must use inRange of all 3-channels of the color image.
* &#x20;Apply post-processing such as morphology to enhance the object segmentation.
* &#x20;Use findContours() to detect all the connected objects
* &#x20;Select only the proper contour around the face. (Hint: can use the contour area)
* &#x20;Then, draw the final contour and a box using  drawContours( ),  boundingRect(), rectangle( )
* &#x20;Need to show example results of each process.

### Part 2. Temperature Measurement

#### Temperature from Intensity data

The intensity value of the image is the temperature data scaled within the pre-defined temperature range. Use the intensity value to estimate the temperature.

![](<../../.gitbook/assets/image (92).png>)

* &#x20;Analyze the intensity values(grayscale, 0-255) of the given image.
* &#x20;The actual temperature for this lab is ranged from 25(I=0) to 40 C (I=255). &#x20;
* &#x20;Estimate the (1) maximum temperature and (2) average tempearture within ONLY the segmented area (Contour Area)
* &#x20;For average tempeature, use the data within the Top 5% of the tempeature in Descending order.
  * &#x20;Hint:  cv∷sort( )  in SORT\_DESCENDING
* &#x20;Show the result as TEXT on the final output image.
  * Hint:  cv∷putText( )
* &#x20;Your final output should be similar to result of the the Demo\_Video.

## III. Report and Demo Video

You are required to write a consice lab report and submit the program files and the demo video.

Lab Report:

* &#x20;Show what you have done with concise explanations and example results of each necessary process
* &#x20;In the appendix, show your source code.
* &#x20;Submit in both PDF and original file (\*.docx etc)
* &#x20;No need to print out. Only the On-Line submission.

&#x20;Demo Video:

* &#x20;Create a demo video with a title page showing the course name, data and your names
* &#x20;Submit in Hisnet

Source Code:

* &#x20;Zip all the necessary source files.
* Only the source code files. Do not submit image files, project files etc.
