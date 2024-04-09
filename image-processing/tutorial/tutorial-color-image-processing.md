# Tutorial: Color Image Processing

## Tutorial: Color Image Processing

​

Tutorial: Color Image Segmentation

## I. Introduction

In this tutorial, you are learn how to segment a colored object using a web-cam. We will use inRange() algorithm to segment the moving colored object and draw contour or boundary boxes around the tracked object.

![](https://user-images.githubusercontent.com/38373000/154498244-a570d6a0-23bc-4f2e-abe4-6351be56b491.png)

## II. Tutorial

### OpenCV: inRange()

First, read the OpenCV documentation [read here](https://docs.opencv.org/3.4/d2/de8/group\_\_core\_\_array.html#ga48af0ab51e36436c5d04340e036ce981)

```
void cv::inRange	(	
    InputArray 	src,
    InputArray 	lowerb,
    InputArray 	upperb,
    OutputArray 	dst 
)	
    
// # python   
    dst= cv2.inRange(src, lowerb, upperb, dst=None)
	dst1 = cv2.inRange(src, (0, 128, 0), (100, 255, 100))
```

Parameters

* src first input array.
* lowerb inclusive lower boundary array or a scalar.
* upperb inclusive upper boundary array or a scalar.
* dst output array of the same size as src and CV\_8U type.

### Example: Color Segmentation

One way of choosing the appropriate InRange conditions is analyzing the pixel statistic of a small sub-window within the targeted colored area.

Download the tutorial source code file

* [tutorial code](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Color/Tutorial\_ColorSegmentation\_trackbar\_2021.cpp)
* [test image](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Color/color\_ball.jpg)

Choose the target window to analyze with mouse click and drag.

> See Appendix for the MouseEvent code

Analyze for the standard deviation and mean of the targeted color within the window.

```
  Mat roi_RGB(image, selection);           // Set ROI by the selection box       
  Mat roi_HSV;  
  cvtColor(roi_RGB, roi_HSV, CV_BGR2HSV);  
  Scalar means, stddev;  meanStdDev(roi_HSV, means, stddev);  
  cout << "\n  Selected ROI Means= " <<  means << " \n  stddev= " <<  stddev;  
```

Add slidebars to change the InRange values of each R, G, B or H, S, V and segment each colored ball.

```
/// set dst as the output of InRange
inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
	Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), dst);
```

Apply appropriate morphology (i.e. dilation/erosion/fill holes) to the output binary images to cluster the detected objects into meaningful blobs.

Find all contours and select the contour with the largest area

```
Mat image_disp, hsv, hue, mask, dst;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
…
findContours(dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
/// Find the Contour with the largest area ///
int idx = 0, largestComp = 0;
double maxArea = 0;
for (; idx >= 0; idx = hierarchy[idx][0])
{
	const vector<Point>& c = contours[idx];
	double area = fabs(contourArea(Mat(c)));		
	if (area > maxArea)
	{
		maxArea = area;
		largestComp = idx;
	}
}
```

Draw the contour and a box over the target object

```
/// Draw the Contour Box on Original Image ///
drawContours(image_disp, contours, largestComp, Scalar(255, 255, 255), 4, 8, hierarchy);
Rect boxPoint = boundingRect(contours[largestComp]);
rectangle(image_disp, boxPoint, Scalar(255, 0, 255), 3);
```

Now, segment other color balls

## III. Exercise

### Drawing the trajectory of a colored object with Webcam

Modify your tutorial program to keep drawing the contour on a white or black background image.

* Use your webcam for the source image data.
* Use any colored object as the target.
* Draw rectangles or circles for the output display.

![](https://user-images.githubusercontent.com/38373000/154500186-80fb5560-3c3d-455c-96fc-3eed044835ec.png)

{% embed url="https://www.youtube.com/watch?v=TkyMSWmKRxQ" %}

## Appendix

**Sample code: On Mouse Event**

```
/// On mouse event 
static void onMouse(int event, int x, int y, int, void*)
{
   if (selectObject)  // for any mouse motion
   {
   	selection.x = MIN(x, origin.x);
   	selection.y = MIN(y, origin.y);
   	selection.width = abs(x - origin.x) + 1;
   	selection.height = abs(y - origin.y) + 1;
   	selection &= Rect(0, 0, image.cols, image.rows);  
// Bitwise AND  check selectin is within the image coordinate
   }

   switch (event)
   {
   case CV_EVENT_LBUTTONDOWN:
   	selectObject = true;
   	origin = Point(x, y);
   	break;
   case CV_EVENT_LBUTTONUP:
   	selectObject = false;
   	if (selection.area())
   		trackObject = true;
   	break;
   }
}
```
