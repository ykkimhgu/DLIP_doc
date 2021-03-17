# Tutorial: Thresholding and Morphology

## **Tutorial: Thresholding and Morphology**

## **I. Introduction**

In this tutorial, we will learn how to apply thresholding and morphology algorithms to segment objects from the background. Thresholding is a powerful tool to segment object images into regions or from the background based on the image intensity values. After applying thresholding methods, morphology methods are usually applied for the post-processing such as pruning unwanted spikes, filling holes and connecting broken pieces.  ****Also, you will learn how to draw and analyze the histogram of a digital image to determine the contrast of the image intensity and use this information to balance the contrast and to determine an optimal value for the thresholding. 

## **II. Tutorial**

### **Part 1. Binary Thresholding** 

#### **Local Thresholding Algorithms**

#### **A. Basic Global thresholding**

![](https://lh3.googleusercontent.com/4YB1b61D99qCQW2tBSFXFEDQEOJDcjJ1jSFlQ2QGpk84yVN_YtmC1cgpuEB2BN1MrzlguJdzPrc97xUsaP43n58HdorNlfPIXcqa3iga0DQl0zkzW1OCSaedoolBjKn0iE4Er5c)

#### **B. Optimum Global thresholding**

![](https://lh4.googleusercontent.com/Q9Doe8K-IJgBCvg6EWBbcqCJG-i7nxPOnVKSKI3dh92N7E753FgmQOrwQCx8N65QDmarix8DKAZlr0o7UNnbGbFHdIZZ0QUIoUC6pSRDnzUuP1CsOAkwnrX2maKgFgSQsH4WfFw)

#### \*\*\*\*

#### **Thresholding Application: OpenCV**  

This tutorial program uses a trackbar to select the threshold value manually. Apply various morphology processes to given images**.**  


1.Download ‘Thresholding\_OpenCV\_Demo.cpp’ and test images.  Apply different values of thresholding for each images

* source code: [click here to download](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/threshold_demo.cpp)
* test images: [click here to download](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial_Threshold_Morp/images)

**2.** Modify the tutorial program to include ‘Otsu method’. Compare the global thresholding and Otsu method results

3. Modify the tutorial program to plot ‘histogram’ of the image.  

* Use this tutorial for help: [click here](https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html)

**4.** Apply ‘Local Adaptive Thresholding’  on the following images. Compare the results of the global thresholding.

![](../../.gitbook/assets/image%20%2882%29.png)



**OpenCV: Global and Optimal Thresholding**

{% embed url="https://docs.opencv.org/3.4/d7/d1b/group\_\_imgproc\_\_misc.html\#gae8a4a146d1ca78c626a53577199e9c57" %}

![](../../.gitbook/assets/image%20%2843%29.png)

* Sample code

{% tabs %}
{% tab title="C++" %}
```cpp
int threshold_value = 0;
int threshold_type = 3;
int morphology_type = 0;

int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst, dst_morph;
src = imread("Finger_print_gray.tif", 1);

/* threshold_type
0: Binary
1: Binary Inverted
2: Threshold Truncated
3: Threshold to Zero
4: Threshold to Zero Inverted*/

threshold(src, dst, threshold_value, max_BINARY_value, threshold_type);

```
{% endtab %}

{% tab title="Python" %}
```python
img = cv.imread('gradient.png',0)
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

```
{% endtab %}
{% endtabs %}



### **Part 2. Morphology** 

#### **Morphology Application: OpenCV**  

This tutorial program uses a trackbar to select the threshold value manually. Apply various morphology processes to given images.  
  
**1.** Refer to sample code below. Apply several morphology to obtain clear segmentation of the object in given images. 

* For Python tutorial: [click here](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html) 

2. Apply several morphology to obtain clear segmentation of the object in given images. 

3. Explain which morphology process you have used and explain the reason. ****

\*\*\*\*

* Sample code

{% tabs %}
{% tab title="C++" %}
```cpp
// Structuring Element
int shape = MORPH_RECT; // MORPH_CROSS, MORPH_ELLIPSE  
  Mat element = getStructuringElement( shape, Size(M, N) )                                     
 
// Apply a morphology operation
dilate( src, dst_morph, element);
erode ( src, dst_morph, element);
```
{% endtab %}

{% tab title="Python" %}
```python
import cv2 as cv
import numpy as np
img = cv.imread('j.png',0)
cv.getStructuringElement(cv.MORPH_RECT,(5,5))
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)

```
{% endtab %}
{% endtabs %}



## **III. Exercise**

After applying thresholding and morphology, we can identify and extract the target objects from the background by finding the contours around the connected pixels. 

**Goal: Count the number of water bubbles for a thermal fluid experiment**

![&amp;lt;img&amp;gt;Bluerred image.](https://lh4.googleusercontent.com/2OZKpPmzK6SzQUEPrkNzsmuFTFf8D_bTq-GXZ2Uqr5OLe-JKL1vQnYkSZU3gMKcOgIw64qv3CcfZu2974nTxWJDQSKzEbqHCz4FpWqUEhT5kh4Eg0E_4B42QfGvpOzNU4C5OtwI)

* Analyze the histogram of the image. 
* Apply a filter to remove image noises
* Choose the appropriate threshold value.
* Apply the appropriate morphology method to segment bubbles
* Find the contour and draw the segmented objects.
* Exclude the contours which are too small or too big to be a bubble
* Count the number of bubbles



#### **Tip: \(contour\_demo.cpp\)**

```cpp
// example code
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
       printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength( contours[i], true ) );       
  }
```



