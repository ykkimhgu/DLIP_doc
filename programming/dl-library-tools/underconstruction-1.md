# Anaconda

## CONDA Installation

[Download the installer on window](https://www.anaconda.com/products/individual#Downloads)

Double click the installer to launch.

Select an install for “Just Me” unless you’re installing for all users \(which requires Windows Administrator privileges\) and click Next.

Select a destination folder to install Anaconda and click the Next button. 

> Install Anaconda to a directory path that does not contain spaces or unicode characters.

Note: Do not install as Administrator unless admin privileges are required  


Do NOT  adding Anaconda to the PATH environment variable

Check  to register Anaconda as your default Python.

Click the Install button.

![](../../.gitbook/assets/image%20%28312%29.png)

Click the Install button. If you want to watch the packages Anaconda is installing, click Show Details.

Click the Next button.

After a successful installation you will see the “Thanks for installing Anaconda” dialog box:



![../../../\_images/win-install-complete.png](https://docs.anaconda.com/_images/win-install-complete.png)



Start ' Anaconda Navigator'

![](../../.gitbook/assets/image%20%28313%29.png)

### 

### Installing Python \(3.7\) in virtual Environment

How to create virtual environment using Conda \(you can skip this now\)

```c
conda create -n py37tf23 python=3.7
```

![](../../.gitbook/assets/image%20%28311%29.png)

### 

### Reference:

{% embed url="https://docs.anaconda.com/anaconda/install/windows/" %}

## CONDA Cheat Sheet

[https://docs.conda.io/projects/conda/en/4.6.0/\_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

{% embed url="https://docs.conda.io/projects/conda/en/4.6.0/\_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf" caption="" %}

## FAQ

### Entry Point Not Error when Opening Conda Env.

FoundPyWinObject\_FromULARGE\_INTEGER@@YAPEAU\_object@@AEBT\_ULARGE\_INTEGER@@[@z](https://github.com/z) could not be located in the dynamic link library C:WINDOWS\SYSTEM32\pythoncom37.dll

**Solution** Delete 'pythoncom37.dll'

