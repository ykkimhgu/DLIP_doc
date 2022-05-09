# Install and Setup

## Installing OpenCV 3.4.13 with Visual Studio 2019

OpenCV Download link: [https://opencv.org/releases/](https://opencv.org/releases/)

![](<../../../.gitbook/assets/image (1).png>)

Opencv 설치 파일 다운로드 >> 설치 파일 압축풀기

**C:\ opencv-3.4.13 폴더** 새롭게 생성 >> 아래 그림과 같이 설치파일 복사

> C:\ opencv-3.4.13 \build, C:\ opencv-3.4.13 \sources\\

![](<../../../.gitbook/assets/image (69).png>)

### **PATH 환경설정**

**제어판> 시스템 및 보안 > 시스템 > 고급 시스템 설정 – 고급 탭 > 환경변수(N)** 클릭

![](<../../../.gitbook/assets/image (26).png>)

**‘시스템 변수’** 항목에 있는 변수 명 중 ‘\*\*Path’\*\*를 찾아 **편집**

![image](https://user-images.githubusercontent.com/38373000/156104553-0419d61a-4eb9-4a1b-864d-33fa093108a7.png)

![](<../../../.gitbook/assets/image (14).png>)

**새로 만들기** > **찾아보기** > **C:\opencv-3.4.13\build\x64\vc15\bin 선택**

> OpenCV dll 파일 경로 설정과정임

![image](https://user-images.githubusercontent.com/38373000/156104789-1ef8b2d1-1b7a-432b-832a-540e150ac75c.png)

Path 설정 완료되면 컴퓨터 재부팅

## OpenCV 프로젝트 속성시트(Property sheet)만들기

Visual Studio 201x 실행 후

* 방법 1) **새 프로젝트 만들기** **> 빈 프로젝트**

![image](https://user-images.githubusercontent.com/38373000/156105330-702e389a-9bd0-46bd-a617-2510e68b487b.png)

* 방법 2) **파일> 새로 만들기> 프로젝트 선택**

\*\*빈 프로젝트: 프로젝트 이름 \*\* `OpenCVprop` 입력 > **만들기**

![](<../../../.gitbook/assets/image (22).png>)

###

### **Debug x64 Property Sheet 만들기**

**메뉴: 보기>다른 창>속성 관리자** 선택

**속성 관리자 창 > OpenCVprop > Debug|x64** 위에 RightClick.

\*\*새 프로젝트 속성 시트 추가 > 이름 \*\*: `opencv-3.4.13_debug_x64.props` 으로 지정 > **추가**

> 반드시 이름에 .props까지 기재할 것
>
> Debug | x64 에서 설정.

![](<../../../.gitbook/assets/image (35).png>)

![](<../../../.gitbook/assets/image (15).png>)

***

\*\*속성관리자 창: \*\* **Debug | x64** > `opencv-3.4.13_debug_x64` double click

![](<../../../.gitbook/assets/image (75).png>)

**\[Opencv 헤더 (include) 디렉터리 추가]**

**공용 속성 > C/C++ > 일반 > 추가 포함 디렉터리 > 편집**

![image](https://user-images.githubusercontent.com/38373000/156106499-3fc82297-8d5f-4e6b-9cf1-0de4c10fda62.png)

**> 추가 포함 디렉터리**> **줄추가** : _아래 경로들을 순서에 맞게 추가_ > **확인**

* C:\opencv-3.4.13\build\include
* C:\opencv-3.4.13\build\include\opencv
* C:\opencv-3.4.13\build\include\opencv2

![](<../../../.gitbook/assets/image (55).png>)

**\[ OpenCV lib 디렉터리 추가]**

**공용 속성 > 링커 > 일반 > 추가 라이브러리 디렉터리 > 편집** > **추가 라이브러리 디렉터리**

* `C:\opencv-3.4.13\build\x64\vc15\lib` 추가

![image](https://user-images.githubusercontent.com/38373000/156107129-4f08152a-678d-41b1-a6e7-686cdc5fe966.png) ![](<../../../.gitbook/assets/image (58).png>)

**\[ OpenCV lib 파일 추가]**

**공용 속성> 링커 > 입력 > 추가 종속성>** **편집**

* `opencv_world3413d.lib` 경로 추가

> debug 모드에서는 'd' (xxx3413d.lib) 로 표시된 파일을 추가

![](<../../../.gitbook/assets/image (30).png>)

![](<../../../.gitbook/assets/image (44).png>)

**(중요)**

위 설정 완료 후 반드시 `opencv-3.4.13_debug_x64` 저장

![image](https://user-images.githubusercontent.com/38373000/156107897-60ad7691-cc4b-48de-86d1-acee1b70a332.png)

### **Release x64 Property Sheet 만들기**

**속성 관리자 창 > OpenCVprop** > **Release**|**x64** : RightClick.

**새 프로젝트 속성 시트 추가 > 이름**: `opencv-3.4.13_release_x64.props`으로 지정 후 추가

![](<../../../.gitbook/assets/image (72).png>)

위에 설명한 **Debug x64 Property Sheet** 만들기 과정과 유사하며, 아래 경로를 추가하여 반복

**공용 속성 > C/C++ > 일반 > 추가 포함 디렉터리 >** 경로추가

* C:\opencv-3.4.13\build\include
* C:\opencv-3.4.13\build\include\opencv
* C:\opencv-3.4.13\build\include\opencv2

**공용 속성 > 링커 > 일반 > 추가 라이브러리 디렉터리** > 경로 추가

* C:\opencv-3.4.13\build\x64\vc15\lib

**공용 속성> 링커 > 입력 > 추가 종속성>** 경로추가

* opencv\_world3413.lib

> release에서는 'opencv\_world3413.lib' 로 해야함. ('\*\*\*d.lib)는 아님!!. ', debug 에서만 'opencv\_world3413d.lib' 로 설정해야 함

**(중요)**

위 설정 완료 후 반드시 `opencv-3.4.13_release_x64` 저장

## OpenCV VS프로젝트 만들기

Visual Studio 2019 실행 후 '**파일> 새로 만들기> 프로젝트 선택**

**Visual C++ > 빈 프로젝트** : 프로젝트 이름 **opencv\_simple\_demo** 입력 후 만들기

### Project Property Sheet 설정

**메뉴>보기>다른 창>속성 관리자** 선택

**속성 관리자 창 > 프로젝트명** > **Debugx64** : RightClick.

**'기존 속성 시트 추가**' 선택 후 앞에서 저장된 **Property Sheet** " **opencv-3.4.13\_debug\_x64.props "** 를 추가

동일한 과정 **Release|x64** 항목에서 **Property Sheet** " **opencv-3.4.13\_release\_x64.props "** 를 추가

![](<../../../.gitbook/assets/image (52).png>)

![](<../../../.gitbook/assets/image (57).png>)

### 소스파일 만들기

**보기 – 솔루션탐색기 > \[프로젝트] > 소스 파일 > 추가 > 새항목** click

C++파일(cpp) 선택 후 **opencv\_simple\_demo.cpp 생성**

![](<../../../.gitbook/assets/image (48).png>)

![](<../../../.gitbook/assets/image (45).png>)

구성 관리자를 **Debug x64**로 설정 후 아래 코드를 입력.

디버깅하지 않고 시작 (CTRL+F5)

![](<../../../.gitbook/assets/image (59).png>)

**Demo 코드 1**: Image File Read

* 이미지 파일 다운로드: [Click here](https://github.com/ykkimhgu/DLIP-src/blob/main/tutorial-install/testImage.JPG)

> 이미지 파일과 소스코드가 동일 폴더에 있어야 함!!

```cpp
#include <opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//* @function main
int main()
{
Mat src;

src = imread("testImage.jpg", 1);/// Load an image

if (src.empty())/// Load image check
{
cout << "File Read Failed : src is empty" << endl;
waitKey(0);
}

/// Create a window to display results
namedWindow("DemoWIndow", CV_WINDOW_AUTOSIZE); //CV_WINDOW_AUTOSIZE(1) :Fixed Window, 0: Unfixed window

if (!src.empty())imshow("DemoWIndow", src); // Show image

waitKey(0);//Pause the program
return 0;
}
```

Expected Output

![](<../../../.gitbook/assets/image (66).png>)

***

**Demo 코드 2:** Camera Open and capture

```cpp
#include "opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    VideoCapture cap(0); // open the video camera no. 0

    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
   namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

    while (1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame); // read a new frame from video
         if (!bSuccess) //if not success, break loop
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }
        imshow("MyVideo", frame); //show the frame in "MyVideo" window

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
       {
            cout << "esc key is pressed by user" << endl;
            break; 
       }
    }
    return 0;
}
```

Expected Output

![](<../../../.gitbook/assets/image (81).png>)
