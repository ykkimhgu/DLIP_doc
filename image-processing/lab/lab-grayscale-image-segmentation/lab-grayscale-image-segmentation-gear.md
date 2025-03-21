# LAB: Grayscale Image Segmentation -Gear

## LAB: Grayscale Image Segmentation

## I. Introduction

**Goal**: Defective Gear Inspection System

Plastic gears are widely used in many applications, including toys, RC cars, and plastic-based hardware. Since they are made of plastic, it is fragile and can have broken gear teeth. You are asked to develop a machine vision system that can inspect defective plastic gears.

[Download the test image](https://github.com/ykkimhgu/DLIP-src/blob/main/images/Lab_GrayScale_Gears.zip)

After analyzing histogram, applying thresholding and morphology, we can identify and extract the target objects from the background by finding the contours around the connected pixels.

## II. Procedure

* Design algorithms to detect defective gear for given images.
  * Include a flowchart to explain the algorithm flow.
* You should apply image processing algorithms, which you have learnt in class, as much as possible.
* If you want to use other algorithms that were not covered in class, then you should briefly explain how that algorithm works.
* For the output, you should calculate the following
  * Number of defective teeth
  * Diameter of the gear
  * Quality Inspection (Pass or Fail)
* You must explain each process with appropriate results
* You MUST include all the following in the report.
  * A simple flowchart to explain your algorithm
  * Apply appropriate filters to enhance image
  * Explain how the appropriate threshold value was chosen (if used)
  * Apply the appropriate morphology method to segment parts (if necessary)
    * Find the contour and draw the segmented objects. See Appendix
  * Explain how to determine the defective teeth

### Output Examples

#### Examples: Output images of each process

<figure><img src="../../../.gitbook/assets/image (235).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (347).png" alt=""><figcaption></figcaption></figure>

## III. Report

You are required to write a concise lab report and submit the program files.

First, read LAB Report Instruction

#### Lab Report:

* Use the given Lab Report Template
* Show what you have done with concise explanations and example results of each necessary process
* In the appendix, show your source code.
* You must write the report in markdown file (\*.md),
  * Recommend (Typora 0.x < 1.x) or Notion
  *   When embedding images

      > Option 1) If you are using local path images: You must include local image folder with the report in zip file
      >
      > Option 2) Use online link for images.
* Submit in both PDF and original documentation file/images

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
