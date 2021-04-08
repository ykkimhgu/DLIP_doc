# Tutorial: Edge Line Circle Detection

**Tutorial: Edge, StraightLine, Circle Detection**

## **I. Introduction**

In this tutorial, you will learn how to use OpenCV to detect edges, lines and circles. For an application, you will learn how to find straight lanes using Canny edge detection and Hough transformation algorithms.  

## **II. Tutorial**

**Find edges, straight lines and circle shapes**

### **Part 1. Edge Detection**

**We will learn how to use Canny Edge Algorithm to detect and display edges.**

* **Example code: click here**
*  **Image data: click here**

OpenCV Canny\(\):[ read docs](https://docs.opencv.org/3.4.13/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de)



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

* The Standard Hough Transform \([ HoughLines](http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=houghlines#houghlines)\( \) \) ****
  * It gives you the results of\(θ, rθ\)
* The Probabilistic Hough Line Transform \( [ HoughLinesP](http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp)\(\)  \)
  * A more efficient implementation of the Hough Line Transform. It gives as output of extremes\(end\) points of the detected lines \(x0, y0, x1, y1\) 

```cpp
C++: void HoughLines(InputArray image, OutputArray lines, double rho, double theta, int threshold, double srn=0, double stn=0 )
C++: void HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )
 


·         image – 8-bit, single-channel binary source image. The image may be modified by the function.
·         lines – Output vector of lines. Each line is represented by a 4-element vector   , where   and   are the ending points of each detected line segment.
·         rho – Distance resolution of the accumulator in pixels.
·         theta – Angle resolution of the accumulator in radians.
·         threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes (   ).
·         minLineLength – Minimum line length. Line segments shorter than that are rejected.
·         maxLineGap – Maximum allowed gap between points on the same line to link them.


```



