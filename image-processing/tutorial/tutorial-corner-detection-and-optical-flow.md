# Tutorial: Corner Detection and Optical Flow

## **I. Introduction**

In this tutorial, you will learn how to find corners to track an object using OopenCV. To detect corners, Harris corner detection method will be used. To track an object, optical flow tracking method (Lukas-Kanade-Tomasi) will be used.

## **II. Tutorial**

**Download Test Image Files:** [Image data click here](https://github.com/ykkimhgu/DLIP-src/blob/main/images/cornerdetectionImg.zip)

### Part 1. Corner Detection

We will learn how to use Harris Corner Detection to find corners, which is an important algorithm for the camera calibration that requires to find chessboard corners.

* OpenCV function of Harris corner detection: [read docs](https://docs.opencv.org/4.9.0/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345)

![](https://user-images.githubusercontent.com/38373000/230373249-6392f543-2900-4778-ac4e-2581136a5c33.png)

```c++
// C++: 
      
    
void cv::cornerHarris	(	InputArray 	src,
							OutputArray 	dst,
							int 	blockSize,
							int 	ksize,
							double 	k,
							int 	borderType = BORDER_DEFAULT 
)	
```

**Harris corner detector.**

The function runs the Harris corner detector on the image. Similarly to cornerMinEigenVal and cornerEigenValsAndVecs , for each pixel (x,y) it calculates a 2×2 gradient covariance matrix M(x,y) over a blockSize×blockSize neighborhood. Then, it computes the following characteristic:

dst(x,y)=detM(x,y)−k⋅(trM(x,y))2

Corners in the image can be found as the local maxima of this response map.

**Parameters:**

· **src** – Input single-channel 8-bit or floating-point image.

· **dst** – Image to store the Harris detector responses. It has the type CV\_32FC1 and the same size as src .

· **blockSize** – Neighborhood size (see the details on \[**cornerEigenValsAndVecs()**]\(http://docs.opencv.org/2.4/modules/imgproc/doc/feature\_detection.html?highlight=cornerharris#void cornerEigenValsAndVecs(InputArray src, OutputArray dst, int blockSize, int ksize, int borderType)) ).

· **ksize** – Aperture parameter for the \[**Sobel()**]\(http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#void Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType)) operator.

· **k** – Harris detector free parameter. See the formula below.

· **borderType** – Pixel extrapolation method. See \[**borderInterpolate()**]\(http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#int borderInterpolate(int p, int len, int borderType)) .



**Tutorial Code:**

```cpp
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;
const char* source_window = "Source image";
const char* corners_window = "Corners detected";
void cornerHarris_demo( int, void* );
int main( int argc, char** argv )
{
    src = imread("../../../Image/checkedPattern.png", 1);
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    namedWindow( source_window );
    createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo );
    imshow( source_window, src );
    cornerHarris_demo( 0, 0 );
    waitKey();
    return 0;
}
void cornerHarris_demo( int, void* )
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst = Mat::zeros( src.size(), CV_32FC1 );
    cornerHarris( src_gray, dst, blockSize, apertureSize, k );
    Mat dst_norm, dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }
    namedWindow( corners_window );
    imshow( corners_window, dst_norm_scaled );
}
```

### Part 2. **Optical Flow: Lukas Kanade Optical Flow with pyramids**&#x20;



OpenCV function of Harris corner detection: [read docs](https://docs.opencv.org/4.9.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)

```cpp
// C++: 
      
    

void cv::calcOpticalFlowPyrLK	(	InputArray 	prevImg,
                                    InputArray 	nextImg,
                                    InputArray 	prevPts,
                                    InputOutputArray 	nextPts,
                                    OutputArray 	status,
                                    OutputArray 	err,
                                    Size 	winSize = Size(21, 21),
                                    int 	maxLevel = 3,
                                    TermCriteria 	criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                    int 	flags = 0,
                                    double 	minEigThreshold = 1e-4 
)		
```

Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.

**Tutorial**

![](https://user-images.githubusercontent.com/38373000/230376593-22be8343-f915-40af-a4ad-bf67e52ddd13.png)

1. Declare variables for Optical flow functions such as the previous and current images(gray scaled), point vectors and term criteria

```cpp
// Optical flow control parameters
	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	vector<Point2f> points[2], initialP;
	vector<uchar> status;
	vector<float> err;
			
	Mat gray, prevGray;
	bool bInitialize = true;

```

2. Read the current video image and convert to gray scaled image

```c++
cap >> frame;
frame.copyTo(image);
cvtColor(image, gray, COLOR_BGR2GRAY);
```

​

3. First, initialize the reference features to track. Initialization is run only once, unless the user calls this function again.

```c++
// Finding initial feature points to track 
		if (bInitialize)
		{
			goodFeaturesToTrack(gray, points[0], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[0], subPixWinSize, Size(-1, -1), termcrit);
			initialP = points[0];
			gray.copyTo(prevGray);
			bInitialize = false;
		}

```

4. Run Optical flow and save the results on the current image (gray, points\[1])

```c++
// run  optic flow 
	calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,3, termcrit, 0, 0.001);
```

5. Draw found features and draw lines

```c++
// draw tracked features on the image
		for (int i = 0; i < points[1].size(); i++)
		{				
			
			//line(image, points[0][i], points[1][i], Scalar(255, 255, 0));
			line(image, initialP[i], points[1][i], Scalar(255, 255, 0));
			circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
		}
```

6. Swap current results to the next previous image/points

```c++
		// Save current values as the new prev values
		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);

```

7. Show the result image

```c++
		namedWindow("LK Demo", 1);
imshow("LK Demo", image);
```

**Tutorial Code:** [**see code here**](https://docs.opencv.org/4.9.0/d2/d1d/samples_2cpp_2lkdemo_8cpp-example.html#a24)

```cpp
// Machine VIsion - YKKim 2016.11
// OpticalFlow_Demo.cpp 
// Optical flow demonstration
// Modification of OpenCV tutorial code


#include "opencv.hpp" 
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

#define MOVIE	0
#define CAM		1
#define IMAGE	0


void printHelp()
{
	
	cout << "\n A simple Demo of Lukas-Kanade optical flow ,\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n" << endl;
}


int main()
{
	printHelp();
	Mat image;
		
	#if MOVIE	
		VideoCapture cap;
		Mat frame;
		cap.open("road1.mp4");
		if (!cap.isOpened()){
			cout << " Video not read \n";
			return 1;
		}
		//int rate=cap.get(CV_CAP_PROP_FPS);
		cap.read(frame);
	#endif MOVIE

	#if CAM
		VideoCapture cap;
		Mat frame;
		//VideoCapture capture;	
		cap.open(0);
		if (!cap.isOpened())
		{

			cout << "***Could not initialize capturing...***\n";
			cout << "Current parameter's value: \n";

			return -1;
		}
		cout << "Camera Read OK \n";
		cap >> frame;  // Read from cam

	#endif CAM

	#if IMAGE
		image = imread("Traffic1.jpg");	
	#endif IMAGE
	
	// Optical flow control parameters
	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	vector<Point2f> points[2], initialP;
	vector<uchar> status;
	vector<float> err;
			
	Mat gray, prevGray;
	bool bInitialize = true;
	
	for (;;)
	{
		
		#if CAM | MOVIE
			cap >> frame;
			if (frame.empty())
				break;
			frame.copyTo(image);
		#endif CAM

		cvtColor(image, gray, COLOR_BGR2GRAY);
		
		// Finding initial feature points to track 
		if (bInitialize)
		{
			goodFeaturesToTrack(gray, points[0], MAX_COUNT, 0.01, 10); // , Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[0], subPixWinSize, Size(-1, -1), termcrit);
			initialP = points[0];
			gray.copyTo(prevGray);
			bInitialize = false;
		}

		if (nightMode)
			image = Scalar::all(0);

		
		// run  optic flow 
		calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
			3, termcrit, 0, 0.001);

		// draw tracked features on the image
		for (int i = 0; i < points[1].size(); i++)
		{				
			
			//line(image, points[0][i], points[1][i], Scalar(255, 255, 0));
			line(image, initialP[i], points[1][i], Scalar(255, 255, 0));
			circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
		}

		// Save current values as the new prev values
		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);

		
		namedWindow("LK Demo", 1);
		imshow("LK Demo", image);

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'r':
			bInitialize = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			bInitialize = true;
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		}
	}

	return 0;
}
```
