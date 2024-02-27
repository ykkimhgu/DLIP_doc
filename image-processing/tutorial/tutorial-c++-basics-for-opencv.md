# Tutorial: C++ basics
## Tutorial: C++ basics
Deep Learning Image Processing.
Updated. 2024.2

## I. Introduction
The OpenCV Library has **>2500** algorithms, extensive documentation, and sample code for real-time computer vision. You can see basic information about OpenCV at the following sites,
* Homepage: [https://opencv.org](https://opencv.org)
* Documentation: [https://docs.opencv.org](https://docs.opencv.org)
* Source code: [https://github.com/opencv](https://github.com/opencv)
* Tutorial: [https://docs.opencv.org/master](https://docs.opencv.org/master)
* Books: [https://opencv.org/books](https://opencv.org/books)

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/2edf3297-6380-4a58-a188-4157a15c3e92)

In this tutorial, you will learn fundamental concepts of the C++ language to use the OpenCV API. You will learn namespace, class, C++ syntax to use image reading, writing and displaying.

### OpenCV Example Code
#### Image File Read / Write / Display
```cpp
#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
  /*  read image  */
  String filename1 = "image.jpg";  // class
  Mat img = imread(filename1);  //Mat class
  Mat img_gray = imread("image.jpg", 0);  // read in grayscale
  
  /*  write image  */
  String filename2 = "writeTest.jpg";  // C++ class/syntax (String, cout, cin)
  imwrite(filename2, img);
 
  /*  display image  */
  namedWindow("image", WINDOW_AUTOSIZE);
  imshow("image", img);
  
  namedWindow("image_gray", WINDOW_AUTOSIZE);
  imshow("image_gray", img_gray);
  
  waitKey(0);
}
```

#### Mat Class
1. The image data are in forms of 1D, 2D, 3D arrays with values 0\~255 or 0\~1
2. OpenCV provides the Mat class for operating images

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/57b39eb8-1ad7-4d86-9229-21ff7a7fe2b9)

### C++ for OpenCV
OpenCV is provided in C++, Python, Java. We will learn how to use OpenCV in
1. C++ (general image processing)
2. Python (for Deep learning processing)

For C++, we need to learn
* Basic C++ syntax
* Class
* Overloading, namespace, template
* Reference



## II. Tutorial
### C++ Introduction
C++ is a general-purpose programming language created by Bjarne Stroustrup as an **extension of the C programming language**.
C++ is portable and can be used to develop applications that can be adapted to multiple platforms. You can see basic C++ tutorials in following site,
* [https://www.w3schools.com/cpp/](https://www.w3schools.com/cpp/)
* [https://www.cplusplus.com/doc/tutorial/variables/](https://www.cplusplus.com/doc/tutorial/variables/)

### Project Workspace Setting
1. Create the lecture workspace as **C:\Users\yourID\source\repos**

e.g. **C:\Users\ykkim\source\repos**

2. Create sub-directories such as :

**C:\Users\yourID\source\repos\DLIP**
**C:\Users\yourID\source\repos\DLIP\Tutorial**
**C:\Users\yourID\source\repos\DLIP\Include**
**C:\Users\yourID\source\repos\DLIP\Assignment**
**C:\Users\yourID\source\repos\DLIP\LAB**
**C:\Users\yourID\source\repos\DLIP\Image**
![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/786e5037-d2de-40a8-85d5-3db848ad977c)

### III. Define Function
We will learn how to define function



