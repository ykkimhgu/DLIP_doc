# Tutorial: Edge Line Circle Detection

**Tutorial: Edge, StraightLine, Circle Detection**

## **I. Introduction**

In this tutorial, you will learn how to use OpenCV to detect edges, lines and circles. For an application, you will learn how to find straight lanes using Canny edge detection and Hough transformation algorithms.  

## **II. Tutorial**

Find edges, straight lines and circle shapes

* [Example code: click here](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial_Hough)
* [Image data: click here](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial_Hough)

### **Part 1. Edge Detection**

We will learn how to use Canny Edge Algorithm to detect and display edges.

* OpenCV Canny\(\):[ read docs](https://docs.opencv.org/3.4.13/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de)

```cpp
C++: void Canny(InputArray image, OutputArray edges, double threshold1, double threshold2, int apertureSize=3, bool L2gradient=false )
 

·         image – single-channel 8-bit input image.
·         edges – output edge map; it has the same size and type as image .
·         threshold1 – first threshold for the hysteresis procedure.
·         threshold2 – second threshold for the hysteresis procedure.
·         apertureSize – aperture size for the Sobel() operator.
·         L2gradient – a flag, indicating whether a more accurate  L2 norm   should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L1 norm  is enough ( L2gradient=false ).


```

*  Declare and define variables:

```cpp
  Mat src, src_gray;
  Mat dst, detected_edges;
 
  int edgeThresh = 1;
  int lowThreshold;
  int const max_lowThreshold = 100;
  int ratio = 3;  // a ratio of lower:upper
  int kernel_size = 3; //Sobel Operation
String window_name = "Edge Map";

```

*  Loads the source image:

```cpp
/// Load an image
src = imread( argv[1] );
if( !src.data )
  { return -1; }
```

* Create a matrix of the same type and size of src \(to be dst\), to grayscale

```cpp
dst.create( src.size(), src.type() );
cvtColor( src, src_gray, CV_BGR2GRAY );
```

* Create a window to display the results

```cpp
namedWindow( window_name, CV_WINDOW_AUTOSIZE );
```

* Create a Trackbar for the user to enter the lower threshold for our Canny detector

```cpp
createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
```

* First, we blur the image with a filter of kernel size 3:

```cpp
blur( src_gray, detected_edges, Size(3,3) );
```

*  Second, we apply the OpenCV function [Canny](http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=canny#canny):

```cpp
Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
```

* We fill a dst image with zeros \(meaning the image is completely black\).

```cpp
dst = Scalar::all(0);
```

* Finally, we will use the function [copyTo](http://docs.opencv.org/modules/core/doc/basic_structures.html?highlight=copyto#mat-copyto) to map only the areas of the image that are identified as edges \(on a black background\)

```cpp
src.copyTo( dst, detected_edges);
imshow( window_name, dst );
```



### **Part 2. Line Detection: Hough Transform**

In OpenCV, there are two kinds of Hough Lıne Transform

* The Standard Hough Transform \([ HoughLines\( \)](https://docs.opencv.org/3.4.13/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a) \) ****

  * It gives you the results of\(θ, rθ\)

* The Probabilistic Hough Line Transform \( [ HoughLinesP](http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp)\(\)  \)
  * A more efficient implementation of the Hough Line Transform. It gives as output of extremes\(end\) points of the detected lines \(x0, y0, x1, y1\) 

```cpp
void HoughLines(InputArray image, OutputArray lines, double rho, double theta, int threshold, double srn=0, double stn=0 )
void HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )

/*

·         image – 8-bit, single-channel binary source image. The image may be modified by the function.
          lines-Output vector of lines. Each line is represented by a 2 or 3 element vector (ρ,θ) or (ρ,θ,votes) 
                 ρ is the distance from the coordinate origin (0,0) (top-left corner of the image)
                 θ is the line rotation angle in radians ( 0∼vertical line,π/2∼horizontal line )
                 votes is the value of accumulator.
          lines (HoughLinesP) Output vector of lines. Each line is represented by a 4-element vector (x1,y1,x2,y2), where (x1,y1) and (x2,y2) are the ending points of each detected line segment.
·         lines – Output vector of lines. Each line is represented by a 4-element vector   , where   and   are the ending points of each detected line segment.
·         rho – Distance resolution of the accumulator in pixels.
·         theta – Angle resolution of the accumulator in radians.
·         threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes (   ).
·         minLineLength – Minimum line length. Line segments shorter than that are rejected.
·         maxLineGap – Maximum allowed gap between points on the same line to link them.
*/

```

*  Load an image

```cpp
// Loads an image
	const char* filename = "../images/Lane_test.jpg";
	Mat src = imread(filename, IMREAD_GRAYSCALE);
	
	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		return -1;
	}
```

*  Detect the edges using Canny detector

```cpp
	// Edge detection
	Canny(src, dst, 50, 200, 3);
```

*  Copy edges to the images that will display the results in BGR

  ```cpp
  	// Copy edge results to the images that will display the results in BGR
  	cvtColor(dst, cdst, COLOR_GRAY2BGR);
  	cdstP = cdst.clone();
  ```

*   \(Option 1\) Standard Hough Line Transform

  * First, apply the Hough Transform. Then display the results by drawing the lines.

  Output vector of lines. Each line is represented by a 2 or 3 element vector \(ρ,θ\) or \(ρ,θ,votes\) . ρ is the distance from the coordinate origin \(0,0\) \(top-left corner of the image\). θ is the line rotation angle in radians \( 0∼vertical line,π/2∼horizontal line \). votes is the value of accumulator.  

```cpp
	// (Option 1) Standard Hough Line Transform
	vector<Vec2f> lines;		
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); 
	
		// Draw the detected lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
```

* \(Option 2\) Probabilistic Hough Line Transform
  * Lines \(HoughLinesP\) Output vector of lines. Each line is represented by a 4-element vector \(x1,y1,x2,y2\), where \(x1,y1\) and \(x2,y2\) are the ending points of each detected line segment.

```cpp
vector<Vec4i> linesP; 
	HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); 
	
	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
```

* Show results

```cpp
	// Show results
	imshow("Source", src);
	imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
	imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
```

### **Part 3. Circle Detection: Hough Circles**

Usually, the function detects the centers of circles well but the radius may not be accurate. It helps if you can specify the radius ranges \( minRadius and maxRadius \), if available. Or, you may set maxRadius to a negative number to return centers only without radius search, and find the correct radius using an additional procedure.

* [HoughCircles\(\) OpenCV docs](https://docs.opencv.org/3.4.13/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d)

```cpp
void cv::HoughCircles(InputArray image, OutputArray circles, int method, double dp, double   minDist, double   param1 = 100, double      	param2 = 100, int minRadius = 0, int              	maxRadius = 0  )
 
/*
image:  	8-bit, single-channel, grayscale input image.
Circles: 	Output vector of found circles. Each vector is encoded as 3 or 4 element floating-point vector (x,y,radius) or (x,y,radius,votes) .
method	Detection method, see HoughModes. Currently, the only implemented method is HOUGH_GRADIENT
dp          	Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
minDist   Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
param1 	First method-specific parameter. In case of HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
param2	Second method-specific parameter. In case of HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
minRadius          	Minimum circle radius.
maxRadius         	Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, returns centers without finding the radius. 
*/
```

* Example code

```cpp
vector<Vec3f> circles;
HoughCircles(gray, circles, 3, 2, gray.rows / 4, 200, 100);
for (size_t i = 0; i < circles.size(); i++)
{
	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
 	int radius = cvRound(circles[i][2]);
 	// draw the circle center
 	circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
 	// draw the circle outline
  circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0); 
}
namedWindow("circles", 1);
imshow("circles", src);
```

