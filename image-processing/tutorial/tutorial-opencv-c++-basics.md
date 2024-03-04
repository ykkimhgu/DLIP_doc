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
## Example 1. Read / Write / Display
You can use the OpenCV C++ library to read, write, and display images/videos. Here is a related example.

[Download image](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_OpenCV/image.jpg)
{% tabs %}
{% tab title="DLIP_Tutorial_Image.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  read image  */
	String filename1 = "image.jpg";
	Mat img = imread(filename1);
	Mat img_gray = imread("image.jpg", 0);  // read in grayscale

	/*  write image  */
	String filename2 = "writeTest.jpg";
	imwrite(filename2, img);

	/*  display image  */
	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", img);

	namedWindow("image_gray", WINDOW_AUTOSIZE);
	imshow("image_gray", img_gray);

	waitKey(0);
}
```
{% endtab %}

{% tab title="DLIP_Tutorial_Video.cpp" %}
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
{% tab title="DLIP_Tutorial_Mat_Opeeration.cpp" %}
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
	Mat img = imread("image.jpg");	// CV8UC3 Image
	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	/*  Chnage image type (8UC1 or 32FC1)  */
	Mat img_32F;
	img_gray.convertTo(img_32F, CV_32FC1);
	imshow("img_32F", img_32F);

	//cout << "img_32F: " << img_32F.channels() << endl << endl;

	waitKey(0);
}
```
{% endtab %}
{% endtabs %}

# Basic Image Operation: Crop, Rotate, Resize, Color Convert
The methods for performing tasks such as image crop, rotate, resize, and color conversion (such as converting to grayscale) are as follows. If you want to learn more about the functions below, refer to the OpenCV documentation.

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/25434240-678f-41a1-8364-c33edae8f9e3)

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/244c72f0-86bc-45e2-b349-c4e9e5caf753)

## Example 3. Basic Image Operation

{% tabs %}
{% tab title="DLIP_Tutorial_basic_image_operation.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  read image  */
	Mat img = imread("image.jpg");
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

# Exercise
## Flip horizontally of the original image
* Useful for Webcam operation.
* Implement 'flip' operation on video webcam.
* Make the video horizontal flipped when you press **'h** key.
* Use with the exercise code below to start

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/2dd5d517-8785-4ce8-adcc-13b72245d975)

{% tabs %}
{% tab title="DLIP_Tutorial_flip_exercise_img.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{

	Mat img = imread("image.jpg");
	Mat flipedImg;
	flip(img, flipedImg, 0);


	imshow("image", img);
	imshow("flipedImg", flipedImg);
	waitKey(0);
}
```
{% endtab %}
{% endtabs %}

# Shallow Copy vs Deep Copy
* Compile and run the code below and see what happens
* Before you execute this code, try to understand what it does

{% tabs %}
{% tab title="DLIP_Tutorial_shallow_deep_copy.cpp" %}
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;
 
int main()
{
	Mat src, dst_shallow, dst_deep;
	// read image  
	src = imread("image.jpg", 0);
	
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

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/08651da2-78de-43de-925d-fc37d63d35ba)

# Accessing Pixel value
## Method 1. Accessing using `at<type>(v,u)`
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
print("%d", img.at<Vec3b>(0, 0)[0]);
print("%d", img.at<Vec3b>(0, 0)[1]);
print("%d", img.at<Vec3b>(0, 0)[2]);
```

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/3a30bec1-c7fe-403c-bb64-fecfd6013bb3)

## Method 2. Using Pointer for faster operation





