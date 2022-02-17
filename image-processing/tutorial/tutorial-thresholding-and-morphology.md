# Tutorial: Thresholding and Morphology

Deep Learning Image Processing. 
Updated. 2022.2

# I. Introduction

In this tutorial, we will learn how to apply thresholding and morphology algorithms to segment objects from the background. Thresholding is a powerful tool to segment object images into regions or from the background based on the image intensity values. After applying thresholding methods, morphology methods are usually applied for the post-processing such as pruning unwanted spikes, filling holes and connecting broken pieces. Also, you will learn how to draw and analyze the histogram of a digital image to determine the contrast of the image intensity and use this information to balance the contrast and to determine an optimal value for the thresholding.



# II. Tutorial



## Part 1. Binary Thresholding

This tutorial shows how to create a simple code to apply OpenCV function for local thresholding. 



### Thresholding:  OpenCV


First, read the OpenCV documentation 

{% embed url="https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57" %}

![](<../../.gitbook/assets/image (43).png>)

#### Sample code

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



### Example  1-1.   Select the local threshold value manually. 

Download  the example code and test images. 

* [source code_1: Manual Threshold](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/threshold_demo.cpp)
* [source code_2: Threshold with Trackbar](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/threshold_trackbar.cpp)
* [test images](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/testImage.zip)



Run the code on all test  images.

Apply different values of thresholding for each images and find the best threshold values. 



### Example 1-2. Otsu Threshold

Modify the program to include ‘Otsu method’. 

Apply on each test images and compare the results with global thresholding output.



### Example 1-3. Plot Histogram

Plot histogram for each images in gray-scale and analyze if clear threshold value exists in histogram.

* Use this tutorial code for plotting histogram: [click here](https://docs.opencv.org/3.4/d8/dbc/tutorial\_histogram\_calculation.html)



### Example 1-4. Local Threshold

Apply ‘Local Adaptive Thresholding’  on the following images. Compare the results of the global thresholding.

Refer to `adaptiveThreshold()` [documentation](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)

![](<../../.gitbook/assets/image (82).png>)



#### Sample code

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



## Exercise

Copy all the result images for Example 1~4 and compare them in a report.  Show the results to TA before proceeding to Part 2.

----





## Part 2. Morphology

### Morphology:  OpenCV

First, read the OpenCV documentation.

{% embed url="https://docs.opencv.org/3.4.17/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f" %}



#### Sample code

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



### Example  2-1.   Morphology selection with trackbar

Download  the example code  

* [source code_: Morphology with Trackbar](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/threshold_trackbar.cpp)



Apply several morphology to obtain clear segmentation of the object in given images, after Thresholding.



****



## Exercise

Apply Morphology methods after thresholding images. 

Analyze which methods works best for each images. 

Show the results to TA before finishing this tutorial. 

----

