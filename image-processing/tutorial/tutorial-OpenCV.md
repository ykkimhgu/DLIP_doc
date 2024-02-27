# Tutorial: OpenCV
## Tutorial: OpenCV
Deep Learning Image Processing.
Updated. 2024.2

## I. Introduction
The OpenCV Library has **>2500** algorithms, extensive documentation, and sample code for real-time computer vision. You can see basic information about OpenCV at the following sites,
* Homepage: [https://opencv.org](https://opencv.org)
* Documentation: [https://docs.opencv.org](https://docs.opencv.org)
* Source code: [https://github.com/opencv](https://github.com/opencv)
* Tutorial: [https://docs.opencv.org/master](https://docs.opencv.org/master)
* Books: [https://opencv.org/books](https://opencv.org/books)

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/2edf3297-6380-4a58-a188-4157a15c3e92)

In this tutorial, you will learn fundamental concepts of the C++ language to use the OpenCV API. You will learn namespace, class, C++ syntax to use image reading, writing and displaying.

### OpenCV Example Code
#### Image File Read / Write / Display
```cpp
#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
  /*  read image  */
  String filename1 = "image.jpg";  // class
  Mat img = imread(filename1);  //Mat class
  Mat img_gray = imread("image.jpg", 0);  // read in grayscale
  
  /*  write image  */
  String filename2 = "writeTest.jpg";  // C++ class/syntax (String, cout, cin)
  imwrite(filename2, img);
 
  /*  display image  */
  namedWindow("image", WINDOW_AUTOSIZE);
  imshow("image", img);
  
  namedWindow("image_gray", WINDOW_AUTOSIZE);
  imshow("image_gray", img_gray);
  
  waitKey(0);
}
```

### C++ for OpenCV
OpenCV is provided in C++, Python, Java. We will learn how to use OpenCV in
1. C++ (general image processing)
2. Python (for Deep learning processing)

For C++, we need to learn
* Basic C++ syntax
* Class
* Overloading, namespace, template
* Reference



## II. Tutorial
### C++ Introduction
C++ is a general-purpose programming language created by Bjarne Stroustrup as an **extension of the C programming language**.
C++ is portable and can be used to develop applications that can be adapted to multiple platforms. You can see basic C++ tutorials in following site,
* [https://www.w3schools.com/cpp/](https://www.w3schools.com/cpp/)
* [https://www.cplusplus.com/doc/tutorial/variables/](https://www.cplusplus.com/doc/tutorial/variables/)

### Project Workspace Setting
1. Create the lecture workspace as **C:\Users\yourID\source\repos**

e.g. **C:\Users\ykkim\source\repos**

2. Create sub-directories such as :

**C:\Users\yourID\source\repos\DLIP**
**C:\Users\yourID\source\repos\DLIP\Tutorial**
**C:\Users\yourID\source\repos\DLIP\Include**
**C:\Users\yourID\source\repos\DLIP\Assignment**
**C:\Users\yourID\source\repos\DLIP\LAB**
**C:\Users\yourID\source\repos\DLIP\Image**
![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/786e5037-d2de-40a8-85d5-3db848ad977c)

### Define Function
We will learn how to declare and define functions in the header file

#### Exercise
1. Create header files **"TU_DLIP.h", "TU_DLIP.cpp"** under C:\Users\yourID\source\repos\DLIP\Tutorial\Tutorial_Cpp\

2. Declare the function in the header file(**"TU_DLIP.h"**)

```cpp
int sum(int val1, int val2);
```

3. Define the function in the header file(**"TU_DLIP.cpp"**)

```cpp
int sum(int val1, int val2){...}
```

4. Run the main() in **"DLIP_Tutorial_C++_student.cpp"** and print the sum value.

#### [DLIP_Tutorial_C++_student.cpp](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Cpp/DLIP_Tutorial_C%2B%2B_student.cpp)

```cpp
#include "TU_DLIP.h"

#include <iostream>

int main()
{
	// =============================
	// Exercise 1 :: Define Function
	// =============================

	// Add code here

}
```

#### [TU_DLIP.h](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Cpp/TU_DLIP.h)
```cpp
#ifndef _TU_DLIP_H		// same as "#if !define _TU_DLIP_H" (or #pragma once) 
#define _TU_DLIP_H

#include <iostream>

// =============================
// Exercise 1 :: Define Function
// =============================

// Add code here

#endif // !_TU_DLIP_H
```

#### [TU_DLIP.cpp](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Cpp/TU_DLIP.cpp)
```cpp
#include "TU_DLIP.h"

#include <iostream>

// =============================
// Exercise 1 :: Define Function
// =============================

int sum(int val1, int val2)
{
	// Add code here
}
```

### Class
Class is similar to C structure.
* Structure: Cannot inclue functions. Only variables
* Class: Can include variables, functions definition/declaration, other class
![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/dfe704da-817e-4407-9eae-315074dd64c1)

#### Structure (C language)
```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  char number[20];
  char password[20];
  char name[20];
  int balance;
}Account;
```

#### Class (C++)
```cpp
#inclue <iostream>
using namespace std;

/*  Class Definition  */
class Account{
  public:
    char number[20];
    char password[20];
    char name[20];
    int balance;
    void deposit(int money);  // Can include functions
    void withdraw(int money);  // Can include functions
};

/*  Class Function Definition  */
void Account::deposit(int money){
  balance += money;
}
void Account::withdraw(int money){
  balance -= money;
}
```

#### Constructor
Constructor is **special method** automatically called when an object of a class is created.
1. Use the **same** name as the class, followed by parentheses **()**
2. It is always **public**.
3. It does not have any return values.

```cpp
class MyNum{
  public:
    MyNum();  // Constructor 1
    MyNum(int x);  // Constructor 2

    int num;
};

// Class Constructor 1
MyNum::MyNum(){}

// Class Constructor 2
MyNum::MyNum(int x)
{
  num = x;
}

int main(){
  // Creating object by constructor 1
  MyNum mynum;
  mynum.num = 10;

  // Creating object by constructor 2
  MyNum mynum2(10);
}
```

#### Mat Class
1. The image data are in forms of 1D, 2D, 3D arrays with values 0\~255 or 0\~1
2. OpenCV provides the Mat class for operating images

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/57b39eb8-1ad7-4d86-9229-21ff7a7fe2b9)

##### Example
```cpp
#include "opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
  cv::Mat src, gray, dst;
  src = cv::imread("testImage.jpg");

  if (src.empty())
    std::cout << "src is empty!!" << std::endl;

  // Print result
  std::cout << "is empty? \t: " << src.empty() << std::endl;
  std::cout << "channels \t: " << src.channels() << std::endl;
  std::cout << "row of src \t: " << src.rows << std::endl;
  std::cout << "col of src \t: " << src.cols << std::endl;
  std::cout << "type of src \t: " << src.type() << std::endl;

  cv::namedWindow("src");
  cv::imshow("src", src);

  cv::waitKey(0);
}
```

**Results**

![](https://github.com/ykkimhgu/DLIP_doc/assets/84508106/0dfb7c4d-d174-440d-ac31-71f4c7e94f82)

#### Exercise 2

#### Create a Class 'myNum'
1. Declare a class member named as **myNum** in **DLIP_Tutorial_C++_student.cpp**
  * Constructor : MyNum()
  * Member variables: val1, val2 // integer type
  * Member functions: int sum() // returns the sum of val1 and val2
  * Member functions: void print() // prints values of **val1, val2, and sum**

2. Split the declaration and definitions of this class in **TU_DLIP.h** and **TU_DLIP.cpp**

#### [DLIP_Tutorial_C++_student.cpp](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Cpp/DLIP_Tutorial_C%2B%2B_student.cpp)

```cpp
#include "TU_DLIP.h"

#include <iostream>

int main()
{
	// =============================
	// Exercise 1 :: Define Function
	// =============================

	// Add code here


	// ====================================
	// Exercise 2 :: Create a Class 'myNum'
	// ====================================

	// Add code here

}
```

#### [TU_DLIP.h](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Cpp/TU_DLIP.h)
```cpp
#ifndef _TU_DLIP_H		// same as "#if !define _TU_DLIP_H" (or #pragma once) 
#define _TU_DLIP_H

#include <iostream>

// =============================
// Exercise 1 :: Define Function
// =============================

// Add code here


// ====================================
// Exercise 2 :: Create a Class 'myNum'
// ====================================

class MyNum 
{
	// Add code here
};

#endif // !_TU_DLIP_H
```

#### [TU_DLIP.cpp](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Cpp/TU_DLIP.cpp)
```cpp
#include "TU_DLIP.h"

#include <iostream>

// =============================
// Exercise 1 :: Define Function
// =============================

int sum(int val1, int val2)
{
	// Add code here
}

// ====================================
// Exercise 2 :: Create a Class ¡®myNum¡¯
// ====================================

MyNum::MyNum(int x1, int x2)
{
	// Add code here
}

int MyNum::sum(void)
{
	// Add code here
}

void MyNum::print(void)
{
	// Add code here
}
```

[Solution Code](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Cpp/TU_DLIP_answer.cpp)

### Namespace
A namespace provides a scope to the identifiers (the names of types, functions, variables, etc) inside it.
* Uses **::** as scope resolution operator
* Use **namespace** in order to avoid collision using functions with the same name
  e.g. KimHandong --> Student::KimHandong, TA::KimHandong

#### Method 1) calling specific function(recommended)
```cpp
int main(void){
  project_A::add_value(3, 7);
  project_A::subtract_value(10, 2);
  return 0;
}
```

#### Method 2) calling all function in the namespace
```cpp
using namespace project_A;

int main(void){
  add_value(3, 7);
  subtract_value(10, 2);
  return 0;
}
```

* **std::cout, std::cin, std::endl** are also defined in **iostream**
```cpp
// Method 1
std::cout<<"print this"<<std::endl;

// Method 2
using namespace std
cout<<"print this"<<endl;
```

#### Exercise

#### Create another Class 'myNum'
1. Declare class member variables like this in **DLIP_Tutorial_C++_namespace_student.cpp**: **Constructor / val1 / val2 / val3 / sum / print**
   - val1, val2, val3: member variable of integer type
   - sum(): member function that returns the sum of val1, val2, and val3
   - print(): member function that prints val1, val2, val3, and sum

2. Build

3. Use namespace to identify two classes clearly
   - First **myNum** class: namespace name **proj_A**
   - Second **myNum** class: namespace name **proj_B**

4. Build and compare

[DLIP_Tutorial_C++_namespace_student.cpp](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Cpp/DLIP_Tutorial_C%2B%2B_namespace_student.cpp)
```cpp
#include <iostream>

namespace proj_A
{
	// Add code here
}

namespace proj_B
{
	// Add code here
}


void main()
{
	proj_A::myNum mynum1(1, 2, 3);
	proj_B::myNum mynum2(4, 5, 6);

	mynum1.print();
	mynum2.print();

	system("pause");
}
```
