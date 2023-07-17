# MacOS OpenCV C++ in XCode

## MacOS: OpenCV C++ in XCode

This post will show you how to setup your C++ OpenCV algorithm project in MacOS using Apple’s [Xcode IDE](https://apps.apple.com/us/app/xcode/id497799835?mt=12).

## Getting Started

In this section we’ll setup a new C++ command line tool project in Xcode. Then we’ll add OpenCV to the project. Then we’ll look at three ways of capturing images using OpenCV; reading image from disk, reading frames from a video and capturing frames from the webcam. This is the point you’ll have a chance to process the image. For this post we’ll simply display the image back to the user.

The steps we’ll take:

1. Install Xcode and Create new Xcode project
2. Installing OpenCV
3. Linking OpenCV
4. Reading images from disk
5. Reading video from disk
6. Streaming the camera feed
7. Running the tool from the command line

We’ll making use of command line using Terminal. During this post make sure to have the Terminal app open.

### 1. Install Xcode and Create new Xcode project

Install [Xcode](https://apps.apple.com/us/app/xcode/id497799835?mt=12)

Then, create a new Xcode project. From menu select \***File\*** > \***New\*** > \***Project…\***

![img](https://miro.medium.com/v2/resize:fit:639/1\*OJHAKaS63zgxw3hzHyP4Fg.png)

When prompted “\***Choose a template for your new project:\***” search and select **Command Line Tool** under **macOS** tab. Then click **Next**.

![img](https://miro.medium.com/v2/resize:fit:1050/1\*Lmc4aPTkNsmk70wIAzudxw.png)

Then when prompted “\***Choose options for your project:\***” name the product `MyOpenCVAlgorithm`. For the “\***Language:\***” select **C++**. Then click **Next**.

Xcode will create a very simple command line based app project that just contains a single file names `main.cpp`. Let’s run the tool. From menu select _**Product\*** > **Run\***._

![img](https://miro.medium.com/v2/resize:fit:557/1\*NQYZzOH7Yk2Ategw\_yzyjg.png)

### 2. Installing OpenCV

To install OpenCV we’ll be making use of command line tool [Homebrew](https://brew.sh/). Open Terminal app and run the following command:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Once Homebrew is installed run the following command to install OpenCV:

```
brew install opencv
```

**To select a specific OpenCV version** (e..g Opencv 3)

First, check which versions are available. Then choose the specific version (e.g. opencv@3)

```c++
$ brew search opencv  
$ brew install opencv@3
```

The installation may take a while. Homebrew will download the [OpenCV](https://opencv.org/) code and other code it depends on. Then [Homebrew](https://brew.sh/) will build [OpenCV](https://opencv.org/) for you.

Once the installation is finished you should find the [OpenCV](https://opencv.org/) installation under `/usr/local/Cellar/opencv/<VERSION>`. At the time of writing the latest version of [OpenCV](https://opencv.org/) on [Homebrew](https://brew.sh/) is `4.5.0`.

![img](https://miro.medium.com/v2/resize:fit:522/1\*\_0pUsujDn7QVebpkxtkbiw.png)

OpenCV is now installed in our machines. However OpenCV is still not installed or _linked_ to our Xcode project.

### 3. Linking OpenCV

In this step we will link OpenCV to our `MyOpenCVAlgorithm` command line tool target. Here is what we need to do to link OpenCV:

1. Link the OpenCV library modules (Other Linker Flags)
2. Tell Xcode where the library modules live or rather where to look for them (Library Search Paths)
3. Tell Xcode where to find the public interfaces for the functionality contained within the library (Header Search Paths)

To find out the information we need to supply Xcode settings we’ll make use of `pkg-config` to tell us about it.

First let’s install `pkg-config`. Run the following command:

```
brew install pkg-config
```

Once the installation is complete run the following command:

```
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
```

[OpenCV](https://opencv.org/) installation will already have placed a file in `/usr/local/lib/pkgconfig`.

![img](https://miro.medium.com/v2/resize:fit:617/1\*wQAwXhCDNLnDb7OMTXrv9w.png)

Now we can use `pkg-config` to query the configuration settings for OpenCV. Run the following command:

```
pkg-config --libs-only-l opencv4 | pbcopy
```

The above command outputs the OpenCV libraries list and copies it to our clipboard ready to pasted.

![img](https://miro.medium.com/v2/resize:fit:1050/1\*Yv0SjwqcWSJhZN7gfOCC5g.png)

Next let’s link OpenCV to our proejct. Open or go back to the project in Xcode. Open the project navigator (from menu select \***View\*** > \***Navigators\*** > \***Project\***) then select `MyOpenCVAlgorithm` project configuration file (the one with the blue icon).

Open project navigator on the right hand side pane

![img](https://miro.medium.com/v2/resize:fit:707/1\*etRzny31qiGzc1xetgWd8A.png)

Select the project configuration file from the project navigator pane

![img](https://miro.medium.com/v2/resize:fit:405/1\*7dpBA6DTCyMJF7dicMJJhA.png)

Next select `MyOpenCVAlgorithm` under “_**Targets\***_”.\* Then select \***Build settings\*** tab.

![img](https://miro.medium.com/v2/resize:fit:1050/1\*k45YJ9V\_GYquYP9wHEhncQ.png)

Search “\***Other Linker Flags\***” and the paste the value(⌘V keys) in clipboard on the **Other Linking Flags** settings.

![img](https://miro.medium.com/v2/resize:fit:1050/1\*CyFf-\_xJI9zqT09YqVOSyA.png)

That’s the OpenCV library linked. However for now we have only told Xcode the names of the modules we want to link to but not where to find them. Let’s fix that.

First let’s find out where the OpenCV modules live. Run the following command:

```
pkg-config --libs-only-L opencv4 | cut -c 3- | pbcopy
```

Again we have asked `pkg-config` for the answer and copied the value to the clipboard. Search “\***Library Search Path\***” and paste (⌘V) the value on the clipboard as the value.

Settings Library search paths in Xcode:

![img](https://miro.medium.com/v2/resize:fit:1050/1\*4JUjGbpL-RMLtr3tYES3MQ.png)

So far we have linked the OpenCV code to our command line app. However to consume the code we need tell Xcode where OpenCV’s public interface is–or in C++ the _headers_ files location.

Again let’s ask `pkg-config` for the location. Run the following command:

```
pkg-config --cflags opencv4 | cut -c 3- | pbcopy
```

![img](https://miro.medium.com/v2/resize:fit:674/1\*IbJqoNMFVGmt5IbKH0Wc0A.png)

Next let’s tell Xcode where the header files are. In build settings search “\***Header Search Path\***” and paste in the value from the clipboard.

Setting Header search paths in Xcode:

![img](https://miro.medium.com/v2/resize:fit:1050/1\*cCVISWMYghen2RWzUeXitA.png)

Let’s run the app. From menu select \***Product\*** > \***Run\***. You should encounter an error such as the following:

```
dyld: Library not loaded:
       /usr/local/opt/gcc/lib/gcc/10/libgfortran.5.dylib
    Referenced from: /usr/local/opt/openblas/lib/libopenblas.0.dylib
    Reason: no suitable image found.  Did find:
        /usr/local/opt/gcc/lib/gcc/10/libgfortran.5.dylib: code signature in
...
```

This happens because by default apps built through Xcode have codesigning policies enabled. All the app code including its dependencies must be codesigned by the same developer. Let’s disable that for its dependencies.

Select `MyOpenCVAlgorithm` project configuration file on the project navigator. Then from the main pane select `MyOpenCVAlgorithm` under “\***Targets\***” and then select “\***Signing & Capabilities\***” tab. Scroll to the “\***Hardened Runtime\***” section and remove the whole section.

![img](https://miro.medium.com/v2/resize:fit:1041/1\*o9BWhuXQUaeUAw01X9l3Dg.png)

**Note** [Hardened Runtime](https://developer.apple.com/documentation/security/hardened\_runtime) is intended to protect the runtime integrity of your app. If you intend to distribute you application I would recommend you not to delete this section and invest time and effort in securing your app. For this post the intention is to run it locally for development of you OpenCV C++ algorithm.

### 3. Example Code: load an image

In this step we’ll use OpenCV to load an image and display it to the screen.

Open `main.cpp` and the following line after `#include <iostream>`:

```
#include <opencv2/opencv.hpp>
```

Create `main.cpp`.

Prepare an image file to read: e.g opencv-logo.jpg

We will load and read the image from disk using OpenCV’s `imread` function and store it in OpenCV image object (`cv::Mat image`).

```
int main(int argc, const char * argv[]) {
    if (argc == 3) {
        std::string readType(argv[1]);
        std::string filePath(argv[2]);
        if (readType == "--image") {
            cv::Mat image = cv::imread(filePath); 
		    // process image
		    cv::imshow("Image from disk", image);
		    cv::waitKey();           
        }
    }
    return 0;
}
```

Let’s provide an image to our command line tool. From menu select \***Product\*** > \***Scheme\*** > \***Edit Scheme…\***

![img](https://miro.medium.com/v2/resize:fit:738/1\*tC2GQTupk2ZJjAt-9TFm\_Q.png)

Then select \***Run\*** > \***Argument\*** and add the following argument:

```
--image <PATH_TO_IMAGE>
```

![img](https://miro.medium.com/v2/resize:fit:1050/1\*Gnyqjt4zym1\_RJgY27Ej9w.png)

Run the app by selecting from menu \***Product\*** > \***Run\***.

![img](https://miro.medium.com/v2/resize:fit:1050/1\*Up5Y6iL-YNc3lqyd1Ya6dg.png)

**Note** I added the OpenCV logo image to the project and set the image path using the [Xcode project path variable ](https://xcodebuildsettings.com/#source\_root)`SRCROOT`.

## Reference

[How to develop OpenCV C++ in Xcode by Ajwani](https://anuragajwani.medium.com/how-to-develop-an-opencv-c-algorithm-in-xcode-d676b9aad1b7)
