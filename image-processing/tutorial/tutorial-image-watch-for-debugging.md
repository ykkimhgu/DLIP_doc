# Tutorial: Image Watch for Debugging



Image Watch is a plug-in for Microsoft Visual Studio that lets you to visualize in-memory images ([_cv::Mat_](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html) example) while debugging an application.&#x20;



This can be helpful for tracking down bugs, or for simply understanding what a given piece of code is doing.



## Reference

{% embed url="https://docs.opencv.org/4.x/d4/d14/tutorial_windows_visual_studio_image_watch.html" %}

## Installation

Visual Studio Community

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

## How to use

#### Exercise Code

```c
// Test application for the Visual Studio Image Watch Debugger extension

#include <iostream>                        // std::cout
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/imgcodecs/imgcodecs.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp>     // cv::Canny()

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
  
    Mat input;
    input = imread("testImage.jpg",1);;
    if (input.empty())
    {
      cout << "Image Load Fail!!" << endl;
      return -1
    } 
    cout << "Detecting edges in input image" << endl;
    Mat edges;
    Canny(input, edges, 10, 100);

    return 0;
}
```

#### Process

<figure><img src="../../.gitbook/assets/image (2).png" alt=""><figcaption></figcaption></figure>

If an image has a thumbnail, left-clicking on that image will select it for detailed viewing in the _Image Viewer_ on the right. The viewer lets you pan (drag mouse) and zoom (mouse wheel). It also displays the pixel coordinate and value at the current mouse position.

<figure><img src="https://docs.opencv.org/4.x/viewer.jpg" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>







<figure><img src="../../.gitbook/assets/image (4).png" alt=""><figcaption></figcaption></figure>



Right-click on the _Image Viewer_ to bring up the view context menu and enable Link Views (a check box next to the menu item indicates whether the option is enabled).

![](https://docs.opencv.org/4.x/viewer_context_menu.png)

The Link Views feature keeps the view region fixed when flipping between images of the same size. To see how this works, select the input image from the image list–you should now see the corresponding zoomed-in region in the input image

<figure><img src="../../.gitbook/assets/image (5).png" alt=""><figcaption></figcaption></figure>



<figure><img src="../../.gitbook/assets/image (6).png" alt=""><figcaption></figcaption></figure>
