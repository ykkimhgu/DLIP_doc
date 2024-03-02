# Tutorial: OpenCV (C++) Basics

## Tutorial: OpenCV (C++) Basics

Deep Learning Image Processing. Updated. 2024.3

# Introduction

The OpenCV Library has **>2500** algorithms, extensive documentation, and sample code for real-time computer vision. You can see basic information about OpenCV at the following sites,

* Homepage: [https://opencv.org](https://opencv.org)
* Documentation: [https://docs.opencv.org](https://docs.opencv.org)
* Source code: [https://github.com/opencv](https://github.com/opencv)
* Tutorial: [https://docs.opencv.org/master](https://docs.opencv.org/master)
* Books: [https://opencv.org/books](https://opencv.org/books)

![](https://github.com/ykkimhgu/DLIP\_doc/assets/84508106/2edf3297-6380-4a58-a188-4157a15c3e92)

In this tutorial, you will learn fundamental concepts of the C++ language to use the OpenCV API. You will learn namespace, class, C++ syntax to use image reading, writing and displaying.

# Basic Image Processing
## Example code 1: Read / Write / Display Images
```cpp
#include "opencv.hpp"

/* read image */
Mat img = imread(filename1);
Mat img_gray = imread("image.jpg", 0); // read in grayscale

/* write image */
imwrite(filename2, img);

/* display image */
namedwindow("image", WINDOW_AUTOSIZE);
imshow("image", img);

```
