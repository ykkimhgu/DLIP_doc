---
description: Count nuts and bolts
---

# LAB: Grayscale Image Segmentation - Bolt and Nut

## LAB: Grayscale Image Segmentation

Segment and count each nuts and bolts

## I. Introduction

**Goal**: Count the number of nuts & bolts of each size for smart factory

There are 2 different size bolts and 3 different types of nuts. You are required to segment the object and count each parts

[Download the test image](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB_grayscale/Lab_GrayScale_TestImage.jpg)

* Bolt M5
* Bolt M6
* Square Nut M5
* Hexa Nut M5
* Hexa Nut M6

![](https://raw.githubusercontent.com/ykkimhgu/DLIP-src/main/LAB_grayscale/Lab_GrayScale_TestImage.jpg)

After analyzing histogram, applying thresholding and morphology, we can identify and extract the target objects from the background by finding the contours around the connected pixels.

## II. Procedure

You MUST include all the following in the report. Also, you have to draw a simple flowchart to explain the whole process

* Apply appropriate filters to enhance image
* Explain how the appropriate threshold value was chosen
* Apply the appropriate morphology method to segment parts
* Find the contour and draw the segmented objects.
  * For applying contour, see Appendix
* Count the number of each parts

#### Expected Final Output

![image](https://user-images.githubusercontent.com/38373000/226501321-dcb79a67-fffc-4e8d-94f5-3b12e9868f07.png)



## III. Report

You are required to write a concise lab report and submit the program files.

#### Lab Report:

* Show what you have done with concise explanations and example results of each necessary process
* In the appendix, show your source code.
* You must write the report in markdown file (\*.md),
  * Recommend (Typora 0.x < 1.x)
  *   When embedding images

      > Option 1) If you are using local path images: You must include local image folder with the report in zip file
      >
      > Option 2) Use online link for images.
* Submit in both PDF and original documentation file/images
* No need to print out. Only the On-Line submission.

#### Source Code:

* Zip all the necessary source files.
* Only the source code files. Do not submit visual studio project files etc.

## Appendix

**Tip**: (contour\_demo.cpp)

```cpp
// OpenCV - use findCountours function

C++: void findContours (InputOutputArray image, OutputArrayOfArrays contours, int mode, int method, Point offset=Point())

C++: void drawContours(InputOutputArray image, InputArrayOfArrays contours, int contourIdx, const Scalar& color, int thickness=1, int lineType=8, InputArrayhierarchy=noArray(), int maxLevel=INT_MAX, Point offset=Point() )
```

```cpp
// Example code
// dst: binary image
 vector<vector<Point>> contours;
 vector<Vec4i> hierarchy;

  /// Find contours
 findContours( dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  
 /// Draw all contours excluding holes
 Mat drawing( dst.size(), CV_8U,Scalar(255));
 drawContours( drawing, contours, -1, Scalar(0), CV_FILLED);
   
 cout<<" Number of coins are ="<<contours.size()<<endl;
 
 for( int i = 0; i< contours.size(); i++ )
 {
      printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]),                          arcLength( contours[i], true ) );       
 }
```

##

## LAB Submission Instruction

[Please read LAB Report Instruction ](../lab-report-instruction.md)









