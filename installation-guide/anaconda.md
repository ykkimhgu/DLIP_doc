# Anaconda

## ANACONDA Installation

Go to https://www.anaconda.com/

Click and download the Window Installer. or Click here [Download the installer on window](https://www.anaconda.com/products/individual#Downloads)

![](https://user-images.githubusercontent.com/38373000/162143256-abe37dec-7fc9-4fd8-b0d3-fe6a6ab89d6b.png)

Follow the following steps

* Double click the installer to launch.
* Select an install for “Just Me” (recommended)
*   Select a destination folder to install Anaconda and click the Next button.

    > Install Anaconda to a directory path that does NOT contain spaces and unicode(Korean) characters.
* Do NOT adding Anaconda to the PATH environment variable
* Check to register Anaconda as your default Python.

Click the Install button.

![](<../.gitbook/assets/image (313).png>)

After a successful installation you will see the “Thanks for installing Anaconda” dialog box:

![](https://user-images.githubusercontent.com/38373000/162144231-e72c06a6-b34b-423e-94f9-329090e5fb8a.png)

Start ' Anaconda Navigator'

![](<../.gitbook/assets/image (314).png>)



### Installing Python (3.7) in virtual Environment

How to create virtual environment using Conda (you can skip this now)

```c
conda create -n py37tf23 python=3.7
```

![](<../.gitbook/assets/image (311).png>)

After installation, activate the newly created environment

```c
conda activate py37tf23
```

![](<../.gitbook/assets/image (315).png>)

### Reference:

{% embed url="https://docs.anaconda.com/anaconda/install/windows/" %}

## CONDA Cheat Sheet

[https://docs.conda.io/projects/conda/en/4.6.0/\_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf](https://docs.conda.io/projects/conda/en/4.6.0/\_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

{% embed url="https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf" %}

## FAQ

### Entry Point Not Error when Opening Conda Env.

FoundPyWinObject\_FromULARGE\_INTEGER@@YAPEAU\_object@@AEBT\_ULARGE\_INTEGER@@[@z](https://github.com/z) could not be located in the dynamic link library C:WINDOWS\SYSTEM32\pythoncom37.dll

**Solution** Delete 'pythoncom37.dll'
