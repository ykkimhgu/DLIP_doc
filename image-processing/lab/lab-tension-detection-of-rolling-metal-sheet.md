# LAB: Tension Detection of Rolling Metal Sheet

## I. Introduction

This is a simplified industrial problem for designing a machine vision system that can detect the level of tension in the rolling metal sheet.

The tension in the rolling process can be derived by measuring the curvature level of the metal sheet with the camera.

The surface of the chamber and the metal sheet are both specular reflective that can create virtual objects in the captured images. You need to design a series of machine vision algorithms to clearly detect the edge of the metal sheet and derive the curvature and tension level.

![](https://github.com/ykkimhgu/DLIP\_doc/assets/84508106/fc1002ef-e283-4b91-a3ce-b480244cb2b0)

### Problem Conditions

* Use Python OpenCV (\*.py)
* Don't use Chat-GPT or any other online materials/search
* Measure the metal sheet tension level from Level 1 to Level 3.
  * Use the minimum y-axis position of the metal sheet curvature
  * Level 1: >250px from the bottom of the image
  * Level 2: 120\~250 px from the bottom of the image
  * Level 3: < 120 px from the bottom of the image
* Display the output on the raw image
  * Tension level: Level 1\~3
  * Score: y-position \[px] of the curvature vertex from the bottom of the image
  * Curvature edge
* Your algorithm will be evaluated on similar test images
*   You can choose either simple images or challenging images

    * Challenging images: You will get up to 15% higher points



### Dataset

Download the test images of

* [Simple dataset](https://github.com/ykkimhgu/DLIP-src/blob/1d4183c65fb41d2ed5bf093c526c7ef49afa2d8e/LAB\_Tension\_Detection\_dataset/Simple\_Dataset.zip)
* [Challenging dataset](https://github.com/ykkimhgu/DLIP-src/blob/1d4183c65fb41d2ed5bf093c526c7ef49afa2d8e/LAB\_Tension\_Detection\_dataset/Challenging\_Dataset.zip)
* [Video](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB_Tension_Detection_dataset/LAB3_Video.mp4)

## II. Procedure

First, understand fully about the design problem.

Design the algorithm flow. You must show the algorithm flowchart or any other methods to clearly show your strategy.

You can follow the basic procedures as follows. You may add more processes if necessary.

#### Download dataset images

<figure><img src="../../.gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

![](C:%5CUsers%5Cykkim%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20240422095536121.png)

#### From the color raw image, cover to a gray-scaled image

* HINT: copper sheet has a reddish surface\\
* You can use `cv.split()` to see individual channel

![](C:%5CUsers%5Cykkim%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20240422095524759.png)

<figure><img src="../../.gitbook/assets/image (3) (1).png" alt=""><figcaption></figcaption></figure>

#### Apply Pre-processing such as filtering

#### Find ROI (region of interest) of the metal sheet from the image

* HINT: Analyze the image area where the metal sheet is located
* For ROI, it does not have to be a rectangle

#### Within the ROI, find the edge of the metal sheet

* HINT: you need to eliminate other objects besides the metal sheet's edge as much as possible

<figure><img src="../../.gitbook/assets/image (4).png" alt=""><figcaption></figcaption></figure>

![](C:%5CUsers%5Cykkim%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20240422095650794.png)

#### Detect and Display the curvature of the metal edge

*   HINT: Find Contour

    ![](C:%5CUsers%5Cykkim%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20240422095500780.png)

<figure><img src="../../.gitbook/assets/image (5).png" alt=""><figcaption></figcaption></figure>

#### Measure the curvature's vertex (minimum point of Y-axis \[px] ) as the tension SCORE .

* Measure the height from the bottom of the image.

#### Detect the tension level from Lv. 1 to Lv. 3

#### Display the Final Output

* Tension level: Level 1\~3
* Score: y-position \[px] of the curvature vertex from the bottom of the image
* Curvature edge overlay

<figure><img src="https://github.com/ykkimhgu/DLIP_doc/assets/84508106/fc1002ef-e283-4b91-a3ce-b480244cb2b0" alt="" width="563"><figcaption></figcaption></figure>

#### Your algorithm will be validated with other similar test object

## III. Report

#### Lab Report:

* Show what you have done with concise explanations and example results of each necessary process
* In the appendix, show your source code.
* Submit in both PDF and original file (\*.md etc)

#### Source Code:

* Zip all the necessary source files.
* Only the source code files. Do not submit image files, project files etc.

####
