---
description: Count nuts and bolts
---

# LAB: Grayscale Image Segmentation

## LAB: Grayscale Image Segmentation

Segment and count each nuts and bolts

## I. Introduction

**Goal**: Count the number of nuts & bolts of each size for smart factory

There are 2 different size bolts and 3 different types of nuts. You are required to segment the object and count each parts

[Download the test image](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB\_grayscale/Lab\_GrayScale\_TestImage.jpg)

* Bolt M5
* Bolt M6
* Square Nut M5
* Hexa Nut M5
* Hexa Nut M6

![](https://raw.githubusercontent.com/ykkimhgu/DLIP-src/main/LAB\_grayscale/Lab\_GrayScale\_TestImage.jpg)

After analyzing histogram, applying thresholding and morphology, we can identify and extract the target objects from the background by finding the contours around the connected pixels.

## II. Procedure

You MUST include all the following in the report. Also, you have to draw a simple flowchart to explain the whole process

* Apply appropriate filters to enhance image
* Explain how the appropriate threshold value was chosen
* Apply the appropriate morphology method to segment parts
* Find the contour and draw the segmented objects.
  * For applying contour, see Appendix
* Count the number of each parts

#### Expected Final Output

<figure><img src="../../.gitbook/assets/image (334).png" alt=""><figcaption><p>Example of Final Output</p></figcaption></figure>



## III. Report

You are required to write a concise lab report and submit the program files.

#### Lab Report:

* Show what you have done with concise explanations and example results of each necessary process
* In the appendix, show your source code.
* You must write the report in markdown file (\*.md),
  * Recommend (Typora 0.x < 1.x)
  *   When embedding images

      > Option 1) If you are using local path images: You must include local image folder with the report in zip file
      >
      > Option 2) Use online link for images.
* Submit in both PDF and original documentation file/images
* No need to print out. Only the On-Line submission.

#### Source Code:

* Zip all the necessary source files.
* Only the source code files. Do not submit visual studio project files etc.

## Appendix

**Tip**: (contour\_demo.cpp)

```cpp
// OpenCV - use findCountours function

C++: void findContours (InputOutputArray image, OutputArrayOfArrays contours, int mode, int method, Point offset=Point())

C++: void drawContours(InputOutputArray image, InputArrayOfArrays contours, int contourIdx, const Scalar& color, int thickness=1, int lineType=8, InputArrayhierarchy=noArray(), int maxLevel=INT_MAX, Point offset=Point() )
```

```cpp
// Example code
// dst: binary image
 vector<vector<Point>> contours;
 vector<Vec4i> hierarchy;

  /// Find contours
 findContours( dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  
 /// Draw all contours excluding holes
 Mat drawing( dst.size(), CV_8U,Scalar(255));
 drawContours( drawing, contours, -1, Scalar(0), CV_FILLED);
   
 cout<<" Number of coins are ="<<contours.size()<<endl;
 
 for( int i = 0; i< contours.size(); i++ )
 {
      printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]),                          arcLength( contours[i], true ) );       
 }
```

##

## Report Submission Instruction

If the submitted program does not run, it will not be graded. 　

1. Please follow the file names and extensions as specified below

> &#x20;If the submitted file name differs even by one character, you will get penalty points.

* File name:&#x20;
  * DLIP\_LAB1\_yourID\_yourName.\*    (\*.cpp, \*.pdf, \*.zip,  etc..)&#x20;
  * Example:  DLIP\_LAB1\_21900123\_GilDongHong.\*  &#x20;
*   Extension:&#x20;

    * Submit all files as one zip file.&#x20;
    * Example:  DLIP\_LAB1\_21900123\_GilDongHong.zip



2. The following files must be included in the zip file.

* Submit all files as one zip file.&#x20;
  * Example:  DLIP\_LAB1\_21900123\_GilDongHong.zip

&#x20; (a) Report&#x20;

* \*.pdf and \*.md files

> if you attached a local image to the md file, please also submit the image used in  .

(b) Program code and data file

* \*.cpp,   \*.h

> Do not submit the entire folder of solutions and projects
>
> NO Submit: \*_.sin, \*_.vcxproj,  Property sheet files 　 etc

(c) Image/data file &#x20;

* (Lab\_GrayScale\_TestImage.jpg)&#x20;

<figure><img src="../../.gitbook/assets/image (235).png" alt=""><figcaption></figcaption></figure>

1. When submitting your program, please specify the imread() path without any additional relative paths.
   * It is not possible for me to identify/modify and build different relative paths for fetching images for each student.
   * Therefore, when submitting the program, please try to build it with the images in the 'project folder' as shown in the example below. Example) src = imread("Lab\_GrayScale\_TestImage.jpg"); (O) src = imread("../../images/Lab\_GrayScale\_TestImage.jpg"); (X) 　
2. Please make the main statement as clean as possible.

* It's not a good idea to do all of your algorithms within the main function.
* Create a function for each possible function and submit the main function as a combined use of them. 　

5. In the comments, please briefly describe what each function does, and if you have an unusual algorithm and implemented the function, please describe it in detail line by line. 　 Please be sure to download the submitted file and check if it works properly in a new project. If your program does not run due to coding issues, it will not be graded.



**※ 반드시 제출파일을 다운로드 받아 새로운 프로젝트에서 정상적으로 구동되는지 확인 바랍니다.※ 제출한 프로그램이 코딩 상의 문제로 실행이 되지 않는 경우, 채점대상에서 제외됩니다**

1\) 제출과제의 통일성을 위해 제출 파일명 및 확장자 형태를 아래와 같이 지정하오니 반드시 엄수하시기 바랍니다. 제출된 파일명이 **문자 하나라도 다를 경우 감점**하겠습니다.   - 파일명: **DLIP\_LAB1\_학번\_성명** (복사하는 것이 가장 확실합니다)              (예시: **DLIP\_LAB1\_21900123\_홍길동**)   - 확장자: **zip** 으로 압축하여 제출   ※ 전년도 과제 감점 사례 : 학번 표기 오류 / 과제번호 표기 오류 / 타 확장자로 압축 등

2\) 제출하는 압축파일 내에는 아래 파일들이 포함되어야 합니다.   - 보고서(.pdf 및 .md 파일 모두 제출 / Typora 파일->내보내기를 통해 pdf 생성 가능)      · md파일에 local image를 첨부한 경우 사용된 image도 제출바랍니다.   - 과제수행에 사용된 프로그램 코드(.c/cpp 및 \*.h)   - 프로그램 빌드를 위한 이미지파일 (Lab\_GrayScale\_TestImage.jpg)   ※ 제출 금지 목록(절대 제출하지 마세요)    a) 과제에서 수행한 솔루션 및 프로젝트 전체 폴더    b) 솔루션 파일(\*.sin), 프로젝트 파일(\*.vcxproj)    c) 속성시트 파일

3\) 프로그램 제출시 imread() 경로를 추가적인 상대경로가 없도록 지정 바랍니다.   - 학생마다 이미지를 불러오기위한 각기 다른 상대경로를 제가 일일이 파악/수정하여 빌드할 수 없습니다.   - 따라서 제출시에는 아래의 예시와 같이 '프로젝트 폴더'에 있는 이미지로 빌드가능하도록 협조 부탁드립니다.       예시) src = imread("Lab\_GrayScale\_TestImage.jpg");   (O)                 src = imread("../../images/Lab\_GrayScale\_TestImage.jpg");  (X)

4\) main 문은 최대한 깔끔하게 작성하여 제출바랍니다.  - 메인함수 내에서 모든 알고리즘을 수행하는 것은 좋지 않습니다.  - 가능한 각 기능별로 함수를 만들고 메인함수는 그것들을 종합적으로 이용한 형태로 제출바랍니다

5\) 주석은 각 함수별로 어떤 기능을 수행하는지 간단히 작성바라며, 본인이 특이한 알고리즘을 구상하여 함수를 구현했을 경우 line by line으로 상세히 기재바랍니다.　**※ 반드시 제출파일을 다운로드 받아 새로운 프로젝트에서 정상적으로 구동되는지 확인 바랍니다.※ 제출한 프로그램이 코딩상의 문제로 실행이 되지 않는 경우, 채점대상에서 제외됩니다.**



