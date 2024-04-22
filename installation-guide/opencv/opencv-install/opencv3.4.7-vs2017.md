# OpenCV3.4.7 VS2017

## Installing OpenCV 3.4.7 with Visual Studio 2017

C: opencv-3.4.7 폴더를 새롭게 만든 후에 opencv 설치 파일 압축풀기

![](<../../../.gitbook/assets/image (34).png>)

#### **윈도우 환경설정**

**윈도우- 내컴퓨터 - 속성 - 고급 시스템 설정 – 고급 탭**의 환경변수 클릭

‘시스템 변수’ 항목에 있는 변수 명 중 ‘Path’를 찾아 더블클릭

![](<../../../.gitbook/assets/image (14).png>)

‘새로 만들기’ 클릭 – 아래 경로입력 – 확인: C:\opencv-3.4.7\build\x64\vc15\bin

Path 설정 완료되면 컴퓨터 재부팅

![](<../../../.gitbook/assets/image (4) (1).png>)

#### OpenCV Visual Studio 프로젝트 설정 및 속성시트(Property sheet)만들기

Visual Studio 201x 실행 후 '**파일> 새로 만들기> 프로젝트 선택**

![](<../../../.gitbook/assets/image (36).png>)

**Visual C++ > 빈 프로젝트** 선택후 ‘\_openCV\_temp’\_라는 이름으로 생성 OpenCVprop

![](<../../../.gitbook/assets/image (19).png>)

**메뉴>보기>다른 창>속성 관리자** 선택

![](<../../../.gitbook/assets/image (33).png>)

**Debug x64 Property Sheet 만들기**

**속성 관리자 창 > 프로젝트명** > **Debugx64**: RightClick.

**새 프로젝트 속성 시트 추가 > 속성시트명**: ‘opencv-3.4.7\_debug\_x64.props’으로 지정 후 추가

> 반드시 .props까지 기재할 것

![](<../../../.gitbook/assets/image (18).png>)

**공용 속성 > C/C++ > 일반 > 추가 포함 디렉터리 > 편집** click

**추가 포함 디렉터리**> 경로추가

* C:\opencv-3.4.7\build\include
* C:\opencv-3.4.7\build\include\opencv
* C:\opencv-3.4.7\build\include\opencv2

![](<../../../.gitbook/assets/image (39).png>)

![](<../../../.gitbook/assets/image (10).png>)

**공용 속성 > 링커 > 일반 > 추가 라이브러리 디렉터리 > 편집** Click

**링커 추가 라이브러리 디렉터> 경**로 추가

* C:\opencv-3.4.7\build\x64\vc15\lib

![](<../../../.gitbook/assets/image (12).png>)

![](<../../../.gitbook/assets/image (2) (1).png>)

**공용 속성> 링커 > 입력 > 추가 종속성> 편집** Click

**링커 추가 종속성**> 경로추가

* opencv\_world347d.lib

![](<../../../.gitbook/assets/image (30).png>)

![](<../../../.gitbook/assets/image (25).png>)

**Release x64 Property Sheet 만들기**

**속성 관리자 창 > 프로젝트명** > **Release** **x64**: RightClick.

**새 프로젝트 속성 시트 추가 > 속성시트명**: ‘opencv-3.4.7\_release\_x64.props’으로 지정 후 추가

![](<../../../.gitbook/assets/image (17).png>)

**공용 속성 > C/C++ > 일반 > 추가 포함 디렉터리 >** 경로추가

* C:\opencv-3.4.7\build\include
* C:\opencv-3.4.7\build\include\opencv
* C:\opencv-3.4.7\build\include\opencv2

**공용 속성 > 링커 > 일반 > 추가 라이브러리 디렉터리 > 경**로 추가

* C:\opencv-3.4.7\build\x64\vc15\lib

**공용 속성> 링커 > 입력 > 추가 종속성>** 경로추가

* opencv\_world347.lib

> release에서는 'xxx347.lib' , debug 에서는 'xxx347d.lib'
