# LAB Report Instruction



## File Submission&#x20;

If the submitted program does not run, it will not be graded. 　



**Please follow the file names  as specified below**

> &#x20;If the submitted file name differs even by one character, you will get penalty points.

* File name:&#x20;
  * DLIP\_LAB1\_yourID\_yourName.\*    (\*.cpp, \*.pdf, \*.zip,  etc..)&#x20;
  * Example:  DLIP\_LAB1\_21900123\_GilDongHong.\*  &#x20;



**The following files must be included in the zip file.**

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

* images must be in the 'project folder' , the same folder as the main source file
* Example)&#x20;
  * src = imread("Lab\_GrayScale\_TestImage.jpg"); (O)&#x20;
  * src = imread("../images/Lab\_GrayScale\_TestImage.jpg"); (X) 　



<figure><img src="../../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

## Comment on Source Code&#x20;

Please make the main() function as clean as possible.

* It's not a good idea to do all of your algorithms within the main function.
* Create a function for each possible function and submit the main function as a combined use of them. 　

5. In the comments, please briefly describe what each function does, and if you have an unusual algorithm and implemented the function, please describe it in detail line by line. 　 Please be sure to download the submitted file and check if it works properly in a new project. If your program does not run due to coding issues, it will not be graded.



**※ 반드시 제출파일을 다운로드 받아 새로운 프로젝트에서 정상적으로 구동되는지 확인 바랍니다.※ 제출한 프로그램이 코딩 상의 문제로 실행이 되지 않는 경우, 채점대상에서 제외됩니다**

1\) 제출과제의 통일성을 위해 제출 파일명 및 확장자 형태를 아래와 같이 지정하오니 반드시 엄수하시기 바랍니다. 제출된 파일명이 **문자 하나라도 다를 경우 감점**하겠습니다.   - 파일명: **DLIP\_LAB1\_학번\_성명** (복사하는 것이 가장 확실합니다)              (예시: **DLIP\_LAB1\_21900123\_홍길동**)   - 확장자: **zip** 으로 압축하여 제출   ※ 전년도 과제 감점 사례 : 학번 표기 오류 / 과제번호 표기 오류 / 타 확장자로 압축 등

2\) 제출하는 압축파일 내에는 아래 파일들이 포함되어야 합니다.   - 보고서(.pdf 및 .md 파일 모두 제출 / Typora 파일->내보내기를 통해 pdf 생성 가능)      · md파일에 local image를 첨부한 경우 사용된 image도 제출바랍니다.   - 과제수행에 사용된 프로그램 코드(.c/cpp 및 \*.h)   - 프로그램 빌드를 위한 이미지파일 (Lab\_GrayScale\_TestImage.jpg)   ※ 제출 금지 목록(절대 제출하지 마세요)    a) 과제에서 수행한 솔루션 및 프로젝트 전체 폴더    b) 솔루션 파일(\*.sin), 프로젝트 파일(\*.vcxproj)    c) 속성시트 파일

3\) 프로그램 제출시 imread() 경로를 추가적인 상대경로가 없도록 지정 바랍니다.   - 학생마다 이미지를 불러오기위한 각기 다른 상대경로를 제가 일일이 파악/수정하여 빌드할 수 없습니다.   - 따라서 제출시에는 아래의 예시와 같이 '프로젝트 폴더'에 있는 이미지로 빌드가능하도록 협조 부탁드립니다.       예시) src = imread("Lab\_GrayScale\_TestImage.jpg");   (O)                 src = imread("../../images/Lab\_GrayScale\_TestImage.jpg");  (X)

4\) main 문은 최대한 깔끔하게 작성하여 제출바랍니다.  - 메인함수 내에서 모든 알고리즘을 수행하는 것은 좋지 않습니다.  - 가능한 각 기능별로 함수를 만들고 메인함수는 그것들을 종합적으로 이용한 형태로 제출바랍니다

5\) 주석은 각 함수별로 어떤 기능을 수행하는지 간단히 작성바라며, 본인이 특이한 알고리즘을 구상하여 함수를 구현했을 경우 line by line으로 상세히 기재바랍니다.　**※ 반드시 제출파일을 다운로드 받아 새로운 프로젝트에서 정상적으로 구동되는지 확인 바랍니다.※ 제출한 프로그램이 코딩상의 문제로 실행이 되지 않는 경우, 채점대상에서 제외됩니다.**
