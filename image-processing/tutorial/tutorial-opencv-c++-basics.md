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

![](https://github.com/ykkimhgu/DLIP\_doc/assets/84508106/57b39eb8-1ad7-4d86-9229-21ff7a7fe2b9)

## Example 2. Matrix Operation: Create / Copy

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/7959071b-8f63-451f-8bdc-92254bb2686a)

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









