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

# Project Workspace Setting

Create the lecture workspace as **C:\Users\yourID\source\repos**

* e.g. `C:\Users\ykkim\source\repos`

Then, create sub-directories such as :

* `C:\Users\yourID\source\repos\DLIP`

* `C:\Users\yourID\source\repos\DLIP\Tutorial`

* `C:\Users\yourID\source\repos\DLIP\Include`

* `C:\Users\yourID\source\repos\DLIP\Assignment`

* `C:\Users\yourID\source\repos\DLIP\LAB`

* `C:\Users\yourID\source\repos\DLIP\Image`
  
* `C:\Users\yourID\source\repos\DLIP\Props`

<figure><img src = "https://github.com/ykkimhgu/DLIP_doc/assets/84508106/a33ed57c-9573-466a-a696-360cd54ced3e"><figcaption></figcaption></figure>

# Basic Image Processing
## Example 1. Read / Write / Display
You can use the OpenCV C++ library to read, write, and display images/videos. Here is a related example.

**You must Read Documentation!!** [link](https://docs.opencv.org/4.9.0/index.html)

0. Configuration OpenCV 4.9.0 debug, release project property sheet. [Link](https://ykkim.gitbook.io/dlip/installation-guide/opencv/opencv-install)

1. Download HGU logo image and rename **HGU\_logo.jpg**
   * Image Link: [HGU\_logo](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_OpenCV/image.jpg)
   * Image Folder: `C:\Users\yourID\source\repos\DLIP\Image\`
   
2. Create a new C++ project in Visual Studio Community
   * Project Name: `DLIP_Tutorial_OpenCV_Image`
   * Project Folder: `C:\Users\yourID\source\repos\DLIP\Tutorial\`

3. Create a new C+ source file
   * File Name: `DLIP_Tutorial_OpenCV_Image.cpp` or `DLIP_Tutorial_OpenCV_Video.cpp`
  
4. Compile and run. 

{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_Image.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  read image  */
	String HGU_logo = "../../../Image/HGU_logo.jpg";
	Mat src = imread(HGU_logo);
	Mat src_gray = imread(HGU_logo, 0);  // read in grayscale

	/*  write image  */
	String fileName = "writeImage.jpg";
	imwrite(fileName, src);

	/*  display image  */
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);

	namedWindow("src_gray", WINDOW_AUTOSIZE);
	imshow("src_gray", src_gray);

	waitKey(0);
}
```
{% endtab %}

{% tab title="DLIP_Tutorial_OpenCV_Video.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  open the video camera no.0  */
	VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", WINDOW_AUTOSIZE);

	while (1)
	{
		Mat frame;

		/*  read a new frame from video  */
		bool bSuccess = cap.read(frame);

		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}
		imshow("MyVideo", frame);

		if (waitKey(30) == 27) // wait for 'ESC' press for 30ms. If 'ESC' is pressed, break loop
		{
			cout << "ESC key is pressed by user\n";
			break;
		}
	}
}
```
{% endtab %}
{% endtabs %}

# Basic Image Container: Mat Class
## Mat Class
The image data are in forms of 1D, 2D, 3D arrays with values 0\~255 or 0\~1

OpenCV provides the Mat class for operating multi-dimensional images

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/41ec96d5-b662-4125-8b1a-224170544a1c)


## Example 2. Matrix Operation: Create / Copy

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/7959071b-8f63-451f-8bdc-92254bb2686a)
{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_Mat_Operation.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  Create or Construct Mat  */
	Mat A(10, 10, CV_8UC3, Scalar::all(155));
	Mat B(A.size(), CV_8UC1);
	Mat C = Mat::zeros(A.size(), CV_8UC1);
	Mat D = Mat::ones(A.size(), CV_32FC1);

	cout << "MAT A: " << A << endl << endl;
	cout << "MAT B: " << B << endl << endl;
	cout << "MAT C: " << C << endl << endl;
	cout << "MAT D: " << D << endl << endl;

	/*  Get size of A (rows, cols)  */
	cout << "Size of A:  " << A.size() << endl;
	cout << "# of Rows of A:  " << A.rows << endl;
	cout << "# of Cols of A:  " << A.cols << endl;
	cout << "# of Channel of A:  " << A.channels() << endl;

	/*  Copy/Clone Mat A to E/F  */
	Mat E, F;
	A.copyTo(E);
	F = A.clone();

	/*  Convert channel  */
	Mat img = imread("../../../Image/HGU_logo.jpg");	// CV8UC3 Image
	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	/*  Chnage image type (8UC1 or 32FC1)  */
	Mat img_32F;
	img_gray.convertTo(img_32F, CV_32FC1, 1.0/255.0);
	imshow("img_32F", img_32F);

	//cout << "img_32F: " << img_32F.channels() << endl << endl;

	waitKey(0);
}
```
{% endtab %}
{% endtabs %}

# Basic Image Operation: Crop, Rotate, Resize, Color Convert
The methods for performing tasks such as image crop, rotate, resize, and color conversion (such as converting to grayscale) are as follows. If you want to learn more about the functions below, refer to the [OpenCV documentation](https://docs.opencv.org/4.9.0/index.html).

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/25434240-678f-41a1-8364-c33edae8f9e3)

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/244c72f0-86bc-45e2-b349-c4e9e5caf753)

## Example 3. Basic Image Operation

{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_basic_image_operation.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  read image  */
	Mat img = imread("../../../Image/HGU_logo.jpg");
	imshow("img", img);

	/*  Crop(Region of Interest)  */
	Rect r(10, 10, 150, 150);	 // (x, y, width, height)
	Mat roiImg = img(r);
	imshow("roiImg", roiImg);

	/*  Rotate  */
	Mat rotImg;
	rotate(img, rotImg, ROTATE_90_CLOCKWISE);
	imshow("rotImg", rotImg);

	/*  Resize  */
	Mat resizedImg;
	resize(img, resizedImg, Size(1000, 100));
	imshow("resizedImg", resizedImg);

	waitKey(0);
}
```
{% endtab %}
{% endtabs %}

# Exercise 1
## Flip horizontally, Rotate, Resize and Crop of the original image
> Everytime you use OpenCV, you must use documentation!!** [link](https://docs.opencv.org/4.9.0/index.html)
> 
For flipping an image, find more details about `cv::flip()`

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/2dd5d517-8785-4ce8-adcc-13b72245d975)

Crop (ROI)
![Image](https://github.com/user-attachments/assets/0252ff49-39ba-44d1-bd5e-ea0b8bd213b5)

Rotate (45deg CW)
![Image](https://github.com/user-attachments/assets/6de74772-2e45-4582-8152-ccf73eb0a7f3)

Resize (by half)
![Image](https://github.com/user-attachments/assets/ed5af4d8-ce68-487d-99fa-c0eec663c883)


0. Configuration OpenCV 4.9.0 debug, release project property sheet. [Link](https://ykkim.gitbook.io/dlip/installation-guide/opencv/opencv-install)

1. Download HGU logo image and rename **HGU\_logo.jpg**
   * Image Link: [HGU\_logo](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_OpenCV/image.jpg)
   * Image Folder: `C:\Users\yourID\source\repos\DLIP\Image\`
   
2. Create a new C++ project in Visual Studio Community
   * Project Name: `DLIP_Tutorial_OpenCV_EX1`
   * Project Folder: `C:\Users\yourID\source\repos\DLIP\Tutorial\`

3. Create a new C+ source file
   * File Name: `DLIP_Tutorial_OpenCV_EX1.cpp`
  
4. Compile and run. 


{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_EX1.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  read src  */
	// Add code here

	/* Flip src image
	// Add code here and show image

	/*  Crop(Region of Interest)  original image */
	// Add code here and show image


	/*  Show source(src) and destination(dst)  */
	// Add code here
	waitKey(0);
}
```
{% endtab %}
{% endtabs %}

# +Extra Exercise 1
The flip function is useful when working with videos. Implement a program that flips the webcam feed horizontally when the `h` key is pressed using `waitKey()` function.
**Hint: flag vs delay time of waitKey**

{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_EX1_extra.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	// Open video camera with index 0
	VideoCapture cap(0);

	// Check if the video camera is opened successfully
	if (!cap.isOpened())
	{
		cout << "Cannot open the video camera\n";
		return -1;
	}

	// Create a window to display the video feed
	namedWindow("MyVideo", WINDOW_AUTOSIZE);

	bool flipHorizontal = false;

	while (true)
	{
		Mat frame;

		// Read a new frame from the video feed
        	bool readSuccess = cap.read(frame);

		// Check if reading the frame was successful
        	if (!readSuccess)
        	{
            		cout << "Cannot find a frame from the video stream\n";
            		break;
        	}

        	// Add code here
        	

        	// Display the frame in the "MyVideo" window
        	imshow("MyVideo", frame);

        	// Wait for 30ms and check if the 'ESC' key is pressed
        	if (waitKey(30) == 27)
        	{
            		cout << "ESC key is pressed by the user\n";
            		break;
        	}
    	}

    	return 0;
}

```
{% endtab %}
{% endtabs %}

# Shallow Copy vs Deep Copy
## Shallow Copy
**Shallow Copy** means copying only the memory addresses in the memory. Since it copies pointers pointing to the same object or data, the original and the copy end up sharing the same data. This can lead to issues, as modifications to one object or array will affect the other as well.

## Deep Copy
**Deep Copy** means creating a copy of an object or data in a new memory space. The original and the copy are independent, having separate memory spaces, so modifications made to one side do not affect the other.

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/eb306257-1813-44f3-afd6-4c8279262f8d)

## Example 4. Shallow_Deep_Copy
* Compile and run the code below and see what happens
* Before you execute this code, try to understand what it does

{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_deep_copy.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;
 
int main()
{
	Mat src, dst_shallow, dst_deep;
	// read image  
	src = imread("../../../Image/HGU_logo.jpg", 0);
	
	/* Shallow Copy */
	dst_shallow = src;
 
	/* Deep Copy */
	src.copyTo(dst_deep);
 
	flip(src, src, 1);
 
	imshow("dst_shallow", dst_shallow);
	imshow("dst_deep", dst_deep);
	waitKey(0);
	return 0;
}
```
{% endtab %}
{% endtabs %}

# Accessing Pixel value
An image is composed of small units called pixels. Each pixel can be considered as the smallest unit of an image. Pixel intensity represents the brightness of a pixel. For grayscale images, pixel intensity ranges from 0 (black) to 255 (white). In color images, each channel (e.g., Red, Green, Blue) has its intensity value.

Rows and columns define an image's structure. Rows represent the vertical direction of the image, and columns represent the horizontal direction. The position of a pixel is denoted as (row, column) or (v, u), where v represents the row index and u represents the column index.

OpenCV provides different methods to access the intensity values of pixels in an image. Two common methods are using `at<type>(v, u)` and using pointers for faster operations.

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/3a30bec1-c7fe-403c-bb64-fecfd6013bb3)

## Method 1. Accessing using `at<type>(v,u)` (Recommended)
```cpp
Mat image= imread(filename);

image.at<uchar>(v,u)= 255;
image.at<float>(v,u)= 0.9;
 
// For an RGB Image
// (option1) Vec3b: 8-bit 3-D image (RGB)
image.at<cv::Vec3b>(v,u)[0]= 255;
image.at<cv::Vec3b>(v,u)[1]= 255;
image.at<cv::Vec3b>(v,u)[2]= 255;

/* Method 1. Accessing using "at<type>(v, u)" */
// For single channel image(Gray-scale)
printf("%d", img_gray.at<uchar>(0, 0));

// For RGB image
printf("%d", img.at<Vec3b>(0, 0)[0]);
printf("%d", img.at<Vec3b>(0, 0)[1]);
printf("%d", img.at<Vec3b>(0, 0)[2]);
```

## Method 2. Using Pointer for faster operation
```cpp
/* Method 2. Accessing Using Pointer */
// Gray Image
int pixel_temp;
for (int v = 0; v < img_gray.rows; v++)
{
	uchar* img_data = img_gray.ptr<uchar>(v);
	for (int u = 0; u < img_gray.cols; u++)
		pixel_temp = img_data[u];
}

//RGB Image
int pixel_temp_r, pixel_temp_g, pixel_temp_b;
int cnt = 0;

for (int v = 0; v < img.rows; v++)
{
	uchar* img_data = img.ptr<uchar>(v);
	for (int u = 0; u < img.cols * img.channels(); u = u+3)
	{
		pixel_temp_r = img_data[u];
		pixel_temp_g = img_data[u+1];
		pixel_temp_b = img_data[u+2];
	}
}
```

## Example 5. Access pixel intensity of Gray-Scale Image(1D image)
{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_Access.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat src;
	// read image  
	src = imread("../../../Image/HGU_logo.jpg", 0);

	int v = src.rows; //행(가로)
	int u = src.cols; //열(세로)

	for (int i = 0; i < v; ++i)
		for (int j = 0; j < u; ++j)
			printf("%d\n", src.at<uchar>(i, j));

	return 0;
}
```
{% endtab %}
{% endtabs %}


# Exercise 2
## Calculate the average intensity value using `at<type>(v,u)`
Calculate the summation of the pixel intensity and calculate the average intensity value. Use `cv::Mat::rows`, `cv::Mat::cols`.

**You must Read Documentation!!** [link](https://docs.opencv.org/4.9.0/index.html)

0. Configuration OpenCV 4.9.0 debug, release project property sheet. [Link](https://ykkim.gitbook.io/dlip/installation-guide/opencv/opencv-install)

1. Download HGU logo image and rename **HGU\_logo.jpg**
   * Image Link: [HGU\_logo](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_OpenCV/image.jpg)
   * Image Folder: `C:\Users\yourID\source\repos\DLIP\Image\`
   
2. Create a new C++ project in Visual Studio Community
   * Project Name: `DLIP_Tutorial_OpenCV_EX2`
   * Project Folder: `C:\Users\yourID\source\repos\DLIP\Tutorial\`

3. Create a new C+ source file
   * File Name: `DLIP_Tutorial_OpenCV_EX2.cpp`
  
4. Compile and run. 

## Result

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/14e47ec7-6896-4e34-9d58-8996d2cb2197)

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/2bbb3e90-5b66-4b98-9e88-52a4b3fbfc73)

{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_EX2.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	// Load the image in gray-scale
	Mat src = imread("../../../Image/HGU_logo.jpg", 0);

	if (src.empty())
    	{
        	cout << "Error: Couldn't open the image." << endl;
        	return -1;
    	}

    	// Calculate the sum of pixel intensities using 'at' function
	double sumIntensity = 0.0;
	for (int i = 0; i < src.rows; i++)
	{
		// Add code here
	} 

    	// Calculate the total number of pixels in the image
    	int pixelCount =  // Add code here

    	// Calculate the average intensity of the image
    	double avgIntensity = // Add code here

    	// Print the results
    	cout << "Sum of intensity: " << sumIntensity << endl;
    	cout << "Number of pixels: " << pixelCount << endl;
    	cout << "Average intensity: " << avgIntensity << endl;

    	// Display the gray-scale image
    	imshow("src", src);
    	waitKey(0);

    	return 0;
}

```
{% endtab %}

{% tab title="DLIP_Tutorial_OpenCV_EX2_Solution.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    // Load the image
    Mat src = imread("../../../Image/HGU_logo.jpg", 0);

    if (src.empty())
    {
        cout << "Error: Couldn't open the image." << endl;
        return -1;
    }


    // Calculate the sum of pixel intensities using 'at' function
    double sumIntensity = 0.0;
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            // Access each pixel in the gray-scale image and add its intensity to the sum
            sumIntensity += src.at<uchar>(i, j);
        }
    }

    // Calculate the total number of pixels in the image
    int pixelCount = src.rows * src.cols;

    // Calculate the average intensity of the image
    double avgIntensity = sumIntensity / pixelCount;

    // Print the results
    cout << "Sum of intensity: " << sumIntensity << endl;
    cout << "Number of pixels: " << pixelCount << endl;
    cout << "Average intensity: " << avgIntensity << endl;

    // Display the gray-scale image
    imshow("src", src);
    waitKey(0);

    return 0;
}

```
{% endtab %}
{% endtabs %}

# Exercise 3
## Intensity Inversion in Grayscale Images
Write a code to invert the colors of this Grayscale image. The resulting image should look like the following. For example, a pixel with an intensity of **100** should become a value of **255 - 100**, which is **155** after the color inversion. Use `Mat::zeros`, `.at<type>(v,u)`

**You must Read Documentation!!** [link](https://docs.opencv.org/4.9.0/index.html)

0. Configuration OpenCV 4.9.0 debug, release project property sheet. [Link](https://ykkim.gitbook.io/dlip/installation-guide/opencv/opencv-install)

1. Download HGU logo image and rename **HGU\_logo.jpg**
   * Image Link: [HGU\_logo](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_OpenCV/image.jpg)
   * Image Folder: `C:\Users\yourID\source\repos\DLIP\Image\`
   
2. Create a new C++ project in Visual Studio Community
   * Project Name: `DLIP_Tutorial_OpenCV_EX3`
   * Project Folder: `C:\Users\yourID\source\repos\DLIP\Tutorial\`

3. Create a new C+ source file
   * File Name: `DLIP_Tutorial_OpenCV_EX3.cpp`
  
4. Compile and run.


## Result

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/50a25aed-d095-4c20-a13c-b21a7946c024)

{% tabs %}
{% tab title="DLIP_Tutorial_OpenCV_EX3.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    // Load the image
    Mat src = imread("../../../Image/HGU_logo.jpg", 0);

    if (src.empty())
    {
        cout << "Error: Couldn't open the image." << endl;
        return -1;
    }

    // Initialize dst with the same size as srcGray
    Mat dst = Mat::zeros(src.size(), src.type());

    // Invert the colors by accessing each pixel
    // Add code here

    // Display the original and inverted images
    imshow("src", src);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}

```
{% endtab %}
{% endtabs %}


# Assignment (1 week)
## Creating a new dataset of training images
In pre-processings of generating datasets of images, we usually maintain the same image size as the original but apply various geometric transformations such as cropping, translation, and resizing.

Write a code to create the following image datasets from the original image (464x480 px). 
* All three output images for this assignment must be the same sized(464x480 px).
* Try to use pixel accesssing to create output images.
* HINT: Use the outputs from the previous Exercises.

![Image](https://github.com/user-attachments/assets/696c8ea5-95af-4985-badc-5e02bd89d195)

Assignment Setup
   * Project Name: `DLIP_Assignment_OpenCV_Basics`
   * Project Folder: `~\DLIP\Assignment\`
   * File Name: `DLIP_Assignment_OpenCV_Basics.cpp`
  
Submit 
* A short report that shows the output images (PDF)
* Source codes: [Use template source file]([link](https://github.com/ykkimhgu/DLIP-src/tree/main/Assignment))
  









