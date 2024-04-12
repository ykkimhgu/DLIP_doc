# Tutorial: Installation for Py OpenCV

## Installation for Visual Studio Code <a href="#installation" id="installation"></a>

> (updated 2022.4)

This installation guide is for programming Python OpenCV. Make sure you install the correct software version as instructed.

> For DLIP Lectures:

* Python >3.9
* Anaconda for Python >3.9 &#x20;
* OpenCV 4.x

### 1. Install Anaconda

**Anaconda** : Python and libraries package installer.

Follow: [How to install Anaconda](https://ykkim.gitbook.io/dlip/installation-guide/anaconda#conda-installation)

###

### 2. Install Python via Anaconda

> Python 3.9 (2022-1)

Python is already installed by installing Anaconda. But, we will make a virtual environment for a specific Python version.

*   Open Anaconda Prompt(admin mode)



    <figure><img src="https://user-images.githubusercontent.com/23421059/169198062-246162fb-1e21-4d63-9377-a50bf75ef060.png" alt=""><figcaption></figcaption></figure>
* First, update conda

```
conda update -n base -c defaults conda
```

![](https://user-images.githubusercontent.com/23421059/169187097-2e482777-fb8b-45c0-b7f6-408073d8b15b.png)

* Then, Create virtual environment for Python 3.9.&#x20;
* Name the $ENV as `py39`. If you are in base, enter `conda activate py39`

```
conda create -n py39 python=3.9.12
```

![image](https://user-images.githubusercontent.com/23421059/169187275-6699f8ee-a4fc-449e-97d5-c087439d0098.png)

* After installation, activate the newly created environment

```
conda activate py39
```

![image](https://user-images.githubusercontent.com/23421059/169187341-0aaa7552-fac3-43fe-9702-66321c67fc06.png)

### 3. Install Libraries&#x20;

#### Install Numpy, OpenCV, Matplot, Jupyter

```
conda activate py39
conda install -c anaconda seaborn jupyter
pip install opencv-python
```

### [4. Install Visual Studio Code](tutorial-installation-for-py-opencv.md#installation)

[How to Install VS Code](../../installation-guide/ide/vscode/#installation)



### [5. Setup Configuration in  Visual Studio Code](../../installation-guide/ide/vscode/python-vscode.md)

[Python in VS Code](https://ykkim.gitbook.io/dlip/installation-guide/ide/vscode/python-vscode)



***

## EXERCISE&#x20;

First, download the test image file: [Click here](https://github.com/ykkimhgu/DLIP-src/blob/main/tutorial-install/testImage.JPG)

> The image file must be in the same folder as the source file

Create a new source file as `TU_OpenCVtest.py`



### Exercise 1

Run python code and submit the final output image

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# a simple numpy test
a = np.array([1,2,3])
print(a*a)

# Load image
img = cv.imread('testImage.jpg')

# Show Image 
cv.imshow('source',img) 
```



### Exercise 2

Run python code and submit the final output image

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()

    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
```

