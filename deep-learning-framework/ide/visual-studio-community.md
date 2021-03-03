# Visual Studio Community

We will use Visual Studio Community for OpenCV C++ . You can download for free if you have a MSN account.

For Installing OpenCV on Visual Studio Community:[ click here](https://ykkim.gitbook.io/dlip/programming/opencv#installing-opencv-3-4-13-with-visual-studio-2019)

{% embed url="https://ykkim.gitbook.io/dlip/programming/opencv\#installing-opencv-3-4-13-with-visual-studio-2019" %}



## How to Install

Download link:  [https://visualstudio.microsoft.com/ko/vs/community/](https://visualstudio.microsoft.com/ko/vs/community/)

Click **"Free Download"** of Visual Studio Community 2019. After downloading, install the program.

![](../../.gitbook/assets/image%20%2831%29.png)

Select "**C++ programming Desktop**" Option.  Optionally,  select others such as "Visual Studio Extension Pack". Then Select  Install button.

![](../../.gitbook/assets/image%20%2827%29.png)

![](../../.gitbook/assets/image%20%2840%29.png)

Visual Studio 2019 실행 후:  MSN 로그인 . 

> 한동대 이메일 ID 로 가입 후 로그인.



### VS 프로젝트 경로 directory  확인하기

VS 실행 후  **메뉴**&gt;'**도구 &gt; 옵션**  선택

![](../../.gitbook/assets/image%20%2838%29.png)



**프로젝트 및 솔루션 &gt; 위치 &gt; 프로젝트 위치:**  자유롭게 설정\(변경하지 않아도 무관함\)

> 프로젝트 위치: 프로젝트 생성시 소스코드를 포함한 프로젝트의 저장 경로를 의미
>
> Default로 VS 프로젝트가 여기 경로에 생성이 됨.

![](../../.gitbook/assets/image%20%2816%29%20%281%29.png)



## VS 프로젝트 만들기

"Hello Handong" 테스트코드 작성하기



Visual Studio 2019 실행 후 **새 프로젝트 만들기 &gt; 빈 프로젝트**

![](../../.gitbook/assets/image%20%2864%29.png)

프로젝트 이름을  **HelloHandong** 으로 설정 후 만들기 

![](../../.gitbook/assets/image%20%2863%29.png)

**보기 – 솔루션탐색기 &gt;  \[프로젝트명\] &gt;  소스 파일 &gt;  추가 &gt;  새항목**   click

**C++ 파일\(cpp\) 선택 후  helloHandong.cpp 파일 생성**

![](../../.gitbook/assets/image%20%2873%29.png)

![](../../.gitbook/assets/image%20%2870%29.png)

아래 소스코드 입력 후 실행 \(CTRL+F5\)

```cpp
#include <stdio.h>

int main()
{
    printf("Hello, Handong!\n");

    return 0;
}
```

Expected Output

![](../../.gitbook/assets/image%20%2871%29.png)



## See also

For Installing OpenCV on Visual Studio Community

{% embed url="https://ykkim.gitbook.io/dlip/programming/opencv\#installing-opencv-3-4-13-with-visual-studio-2019" %}



