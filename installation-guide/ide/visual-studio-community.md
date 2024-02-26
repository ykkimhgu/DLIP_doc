# Visual Studio Community

(updated 03-03) We will use Visual Studio Community for OpenCV C++ . You can download for free if you have a MSN account.

For Installing OpenCV on Visual Studio Community:[ click here](https://ykkim.gitbook.io/dlip/programming/opencv#installing-opencv-3-4-13-with-visual-studio-2019)

{% embed url="https://ykkim.gitbook.io/dlip/programming/opencv#installing-opencv-3-4-13-with-visual-studio-2019" %}

### How to Install

Download link: [https://visualstudio.microsoft.com/ko/vs/community/](https://visualstudio.microsoft.com/ko/vs/community/)

Click **"Free Download"** of Visual Studio Community 2022. After downloading, install the program.

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(31\).png)

Select "**C++ programming Desktop**" Option. Optionally, select others such as "Visual Studio Extension Pack". Then Select Install button.

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(27\).png)

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(40\).png)

Visual Studio 2022 실행 후: MSN 로그인 .

> 한동대 이메일 ID 로 가입 후 로그인.

#### VS 프로젝트 경로 directory 확인하기

VS 실행 후 **메뉴**>'**도구 > 옵션** 선택

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(38\).png)

**프로젝트 및 솔루션 > 위치 > 프로젝트 위치:** 자유롭게 설정(변경하지 않아도 무관함)

> 프로젝트 위치: 프로젝트 생성시 소스코드를 포함한 프로젝트의 저장 경로를 의미
>
> Default로 VS 프로젝트가 여기 경로에 생성이 됨.

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(16\)%20\(1\).png)

### VS 프로젝트 만들기

"Hello Handong" 테스트코드 작성하기

Visual Studio 2022 실행 후 **새 프로젝트 만들기 > 빈 프로젝트**

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(64\).png)

프로젝트 이름을 **HelloHandong** 으로 설정 후 만들기

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(63\).png)

**보기 – 솔루션탐색기 > \[프로젝트명] > 소스 파일 > 추가 > 새항목** click

**C++ 파일(cpp) 선택 후 helloHandong.cpp 파일 생성**

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(73\).png)

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(70\).png)

아래 소스코드 입력 후 실행 (CTRL+F5)

```
#include <stdio.h>

int main()
{
    printf("Hello, Handong!\n");

    return 0;
}
```

Expected Output

![](https://github.com/ykkimhgu/DLIP\_doc/raw/master/.gitbook/assets/image%20\(71\).png)

### See also

For Installing OpenCV on Visual Studio Community

{% embed url="https://ykkim.gitbook.io/dlip/programming/opencv\#installing-opencv-3-4-13-with-visual-studio-2019" %}

