# Tutorial: Thresholding and Morphology

## Tutorial: Thresholding and Morphology

Deep Learning Image Processing. Updated. 2024.3

## I. Introduction

In this tutorial, we will learn how to apply thresholding and morphology algorithms to segment objects from the background. Thresholding is a powerful tool to segment object images into regions or from the background based on the image intensity values. After applying thresholding methods, morphology methods are usually applied for post-processing such as pruning unwanted spikes, filling holes, and connecting broken pieces. Also, you will learn how to draw and analyze the histogram of a digital image to determine the contrast of the image intensity and use this information to balance the contrast and determine an optimal value for the thresholding.

## II. Tutorial

### Part 1. Binary Thresholding

This tutorial shows how to create a simple code to apply the OpenCV function for local thresholding.

#### Thresholding: OpenCV

First, read the OpenCV documentation

{% embed url="https://docs.opencv.org/4.9.0/" %}

![](<../../.gitbook/assets/image (43).png>)

**Sample code**

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
src = imread("coin.jpg", 1);

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
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('coin.jpg',0)

ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```
{% endtab %}
{% endtabs %}

#### Example 1-1. Select the local threshold value manually.

Download the example code and test images.

* [source code\_1: Manual Threshold](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/DLIP_Tutorial_Threshold_demo.cpp)
* [source code\_2: Threshold with Trackbar](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/DLIP_Tutorial_ThresholdMorph_trackbar.cpp)
* [Test images](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/testImage.zip)

Run the code on all test images.

Apply different values of thresholding for each images and find the best threshold values.

#### Example 1-2. Otsu Threshold

Modify the program to include ‘Otsu method’.

* [Read this documentation](https://docs.opencv.org/4.9.0/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a95251923e8e22f368ffa86ba8bce87ff) for THRESH\_OTSU

Apply on each test image and compare the results with global thresholding output.

#### Example 1-3. Plot Histogram

Calculate and Plot the histogram for each image in gray-scale and analyze if a clear threshold value exists in a histogram.

* [Read this documentation](https://docs.opencv.org/4.9.0/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d) for calculating histogram `calcHist()`

```cpp
void cv::calcHist	(	const Mat * 	images,
                        int 	nimages,
                        const int * 	channels,
                        InputArray 	mask,
                        OutputArray 	hist,
                        int 	dims,
                        const int * 	histSize,
                        const float ** 	ranges,
                        bool 	uniform = true,
                        bool 	accumulate = false 
)		
```

```python
# Python:
hist=cv.calcHist(	images, channels, mask, histSize, ranges[, hist[, accumulate]]	)
```

For plotting histogram, you may use the following function.

```cpp
void plotHist(Mat src, string plotname, int width, int height) {
	/// Compute the histograms 
	Mat hist;
	/// Establish the number of bins (for uchar Mat type)
	int histSize = 256;
	/// Set the ranges (for uchar Mat type)
	float range[] = { 0, 256 };

	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	double min_val, max_val;
	cv::minMaxLoc(hist, &min_val, &max_val);
	Mat hist_normed = hist * height / max_val; 
	float bin_w = (float)width / histSize;	
	Mat histImage(height, width, CV_8UC1, Scalar(0));	
	for (int i = 0; i < histSize - 1; i++) {	
		line(histImage,	
			  Point((int)(bin_w * i), height - cvRound(hist_normed.at<float>(i, 0))),			
			  Point((int)(bin_w * (i + 1)), height - cvRound(hist_normed.at<float>(i + 1, 0))),	
			  Scalar(255), 2, 8, 0);											
	}

	imshow(plotname, histImage);
}
```

See here for full example codes

* Example 1: [Histogram of GrayScale Image File](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/DLIP_Tutorial_Histogram_1D_demo_image.cpp)
* Example 2: [Histogram of Color Image File](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/DLIP_Tutorial_Histogram_demo_image.cpp)
* Example 3: [Histogram of Webcam Image](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/DLIP_Tutorial_Histogram_1D_demo_webcam.cpp)

#### Example 1-4. Local Threshold

Apply ‘Local Adaptive Thresholding’ on the following images. Compare the results of the global thresholding.

![](<../../.gitbook/assets/image (82).png>)

Refer to `adaptiveThreshold()` [documentation](https://docs.opencv.org/4.9.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)

```cpp
//adaptiveThreshold()
void cv::adaptiveThreshold	(	InputArray 	src,
                                 OutputArray 	dst,
                                 double 	maxValue,
                                 int 	adaptiveMethod,
                                 int 	thresholdType,
                                 int 	blockSize,
                                 double 	C 
)		
```

```python
#Python:
dst=cv.adaptiveThreshold(	src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
```

**Sample code**

{% tabs %}
{% tab title="C++" %}
```cpp
void cv::adaptiveThreshold	(	
    InputArray 	src,
	OutputArray 	dst,
    double 	maxValue,
    int 	adaptiveMethod,
    int 	thresholdType,
    int 	blockSize,
    double 	C 
)	
    
adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,3, 11);
```
{% endtab %}

{% tab title="Python" %}
```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('sudoku.png',0)
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```
{% endtab %}
{% endtabs %}

###

***

### Part 2. Morphology

#### Morphology: OpenCV

First, read the OpenCV documentation on morphology.

* [morphologyEx()](https://docs.opencv.org/4.9.0/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)
* [erode()](https://docs.opencv.org/4.9.0/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb)

**Sample code**

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

#### Example 2-1. Morphology selection with trackbar

Download the example code

* [source code\_: Morphology with Trackbar](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/DLIP_Tutorial_ThresholdMorph_trackbar.cpp)

Apply several morphology to obtain clear segmentation of the object in given images, after Thresholding.

***

## Exercise

### Exercise 1

Create a new C++ project in Visual Studio Community

* Project Name: `DLIP_Tutorial_Thresholding`
* Project Folder: `C:\Users\yourID\source\repos\DLIP\Tutorial\`
* Source File: `DLIP_Tutorial_Thresholding.cpp`
* [Test images](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/testImage.zip)

1. Analyze Histogram and apply Thresholding methods on given images.
2. Find the optimal threshold method and value for the object segmentation.
3. Show the results to TA before proceeding to Exercise 2.

### Exercise 2

1. Apply Morphology methods after threshold on all test images.
2. Analyze which morphology methods works best for each images for object segmentation.
3. Show the results to TA before finishing this tutorial.

***
