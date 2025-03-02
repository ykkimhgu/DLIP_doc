# OpenCV Install and Setup

## Installing OpenCV 4.9.0 with Visual Studio 2022

OpenCV Download link: [https://opencv.org/releases/](https://opencv.org/releases/)

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/ad1aa16c-0e6e-4cb1-842b-619b440ceb96)

Opencv 설치 파일 다운로드 >> 설치 파일 압축풀기

**C:\ opencv-4.9.0 폴더** 새롭게 생성 >> 아래 그림과 같이 설치파일 복사

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/bfc8b189-1eec-46ef-8a73-c54c4cc49bc1)

### **PATH 환경설정**

**제어판> 시스템 및 보안 > 시스템 > 고급 시스템 설정 – 고급 탭 > 환경변수(N)** 클릭

![](<../../../.gitbook/assets/image (26) (1).png>)

**‘시스템 변수’** 항목에 있는 변수 명 중 **Path**를 찾아 **편집**

![](https://user-images.githubusercontent.com/38373000/156104553-0419d61a-4eb9-4a1b-864d-33fa093108a7.png)

**새로 만들기** > **찾아보기** > **C:\opencv-4.9.0\build\x64\vc16\bin 선택**

> OpenCV dll 파일 경로 설정과정임

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/15717299-1724-4eb5-b318-45eef7dd44bd)

Path 설정 완료되면 컴퓨터 재부팅

## OpenCV 프로젝트 속성시트(Property sheet)만들기

Visual Studio 2022 실행 후

* 방법 1) **새 프로젝트 만들기** **> 빈 프로젝트**

![](https://user-images.githubusercontent.com/38373000/156105330-702e389a-9bd0-46bd-a617-2510e68b487b.png)

* 방법 2) **파일> 새로 만들기> 프로젝트 선택**

**빈 프로젝트: 프로젝트 이름** : `OpenCVprop` 입력 > **만들기**

![](<../../../.gitbook/assets/image (22).png>)

### **Debug x64 Property Sheet 만들기**

**메뉴: 보기>다른 창>속성 관리자** 선택

**속성 관리자 창 > OpenCVprop > Debug|x64** 위에 RightClick.

**새 프로젝트 속성 시트 추가 > 이름**: `opencv-4.9.0_debug_x64.props` 으로 지정 > **추가**

> 반드시 이름에 .props까지 기재할 것
>
> Debug | x64 에서 설정.

![](<../../../.gitbook/assets/image (35).png>)

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/77183a8d-5ba1-4486-b44e-412cfcc1a2bd)

***

**속성관리자 창**: **Debug | x64** > `opencv-3.4.13_debug_x64` double click

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/7f75af32-f023-4588-93bf-33c47f2f0490)

**\[Opencv 헤더 (include) 디렉터리 추가]**

**공용 속성 > C/C++ > 일반 > 추가 포함 디렉터리 > 편집**

![](https://user-images.githubusercontent.com/38373000/156106499-3fc82297-8d5f-4e6b-9cf1-0de4c10fda62.png)

**> 추가 포함 디렉터리**> **줄추가** : _아래 경로들을 순서에 맞게 추가_ > **확인**

* C:\opencv-4.9.0\build\include
* C:\opencv-4.9.0\build\include\opencv2

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/e96efff9-fe51-4a93-b595-36c97e18d10f)

**\[ OpenCV lib 디렉터리 추가]**

**공용 속성 > 링커 > 일반 > 추가 라이브러리 디렉터리 > 편집** > **추가 라이브러리 디렉터리**

* `C:\opencv-4.9.0\build\x64\vc16\lib` 추가

![image](https://user-images.githubusercontent.com/38373000/156107129-4f08152a-678d-41b1-a6e7-686cdc5fe966.png) ![image](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/b3fbbf5a-fa2d-4962-8956-6ac731616035)

**\[ OpenCV lib 파일 추가]**

**공용 속성> 링커 > 입력 > 추가 종속성>** **편집**

* `opencv_world490d.lib` 경로 추가

> debug 모드에서는 'd' (xxx490d.lib) 로 표시된 파일을 추가

![](<../../../.gitbook/assets/image (30).png>)

![image](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/afd82376-97ca-433b-9f36-815df7c27a25)

**(중요)**

위 설정 완료 후 반드시 `opencv-4.9.0_debug_x64` 저장

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/f40a5676-83b5-4e72-8d73-4f3f15f9f8b5)

### **Release x64 Property Sheet 만들기**

**속성 관리자 창 > OpenCVprop** > **Release**|**x64** : RightClick.

**새 프로젝트 속성 시트 추가 > 이름**: `opencv-4.9.0_release_x64.props`으로 지정 후 추가

![](<../../../.gitbook/assets/image (72).png>)

위에 설명한 **Debug x64 Property Sheet** 만들기 과정과 유사하며, 아래 경로를 추가하여 반복

**공용 속성 > C/C++ > 일반 > 추가 포함 디렉터리 >** 경로추가

* C:\opencv-4.9.0\build\include
* C:\opencv-4.9.0\build\include\opencv2

**공용 속성 > 링커 > 일반 > 추가 라이브러리 디렉터리** > 경로 추가

* C:\opencv-4.9.0\build\x64\vc16\lib

**공용 속성> 링커 > 입력 > 추가 종속성>** 경로추가

* opencv\_world490.lib

> release에서는 'opencv\_world490.lib' 로 해야함. ('\*\*\*d.lib)는 아님!!. ', debug 에서만 'opencv\_world490d.lib' 로 설정해야 함

**(중요)**

위 설정 완료 후 반드시 `opencv-4.9.0_release_x64` 저장

##

## Demo Program

Follow the Tutorial: Create OpenCV Project&#x20;

{% content-ref url="../../../image-processing/tutorial/tutorial-create-opencv-project.md" %}
[tutorial-create-opencv-project.md](../../../image-processing/tutorial/tutorial-create-opencv-project.md)
{% endcontent-ref %}



