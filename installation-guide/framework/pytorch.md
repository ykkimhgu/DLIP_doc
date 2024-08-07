# PyTorch

## What is PyTorch

An open source machine learning framework that accelerates the path from research prototyping to production deployment. It’s a Python-based scientific computing package targeted at two sets of audiences:

* A replacement for NumPy to use the power of GPUs
* a deep learning research platform that provides maximum flexibility and speed

{% embed url="https://pytorch.org/features/" caption="" %}

## How to Install

> \(2022.1 :  Use  Pytorch 1.10.x, CUDA=10.2\)

Select your preferences and run the install command. Please ensure that you have **met the prerequisites below \(e.g., numpy\)**

> [**https://pytorch.org/get-started/locally/**](https://pytorch.org/get-started/locally/)



* Install CUDA or  check the installed GPU CUDA version

```text
nvcc --version
```

* Install **Anaconda:** To install Anaconda, you will use the [64-bit graphical installer](https://www.anaconda.com/download/#windows) for Python  3.x.

* Install **Anaconda**. After installation, run **Anaconda Prompt**

* You can use previous virtual environment e.g. py37 , where other necessary tools are installed.

  ```
  conda activate py37
  ```

  

* *(Optional)  Make a new virtual environment e.g torch110. 

  * For new virtual environment, In the \(base\) of Anaconda Prompt, create a new environment.
  * You need to install necessary tools again for this environment.

```text
conda env list
conda create --name torch16
activate torch16
```



* Install  Python, Numpy, Panda and other prerequisite. Also, install necessary IDE \(Jupyter Notebook, Visual Studio Code etc.\)

> Check Python version. Need to have Python 3.x on Windows. `python --version`
>
> Check the list of packages installed in the environment`conda list`



* Select your preferences and run the install command in your environment. The command can be found in  [**https://pytorch.org/get-started/locally/**](https://pytorch.org/get-started/locally/)  _\*\*_

**`conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`**





## Verify installation

From the command line, type:

```text
python
```

then enter the following code:

```text
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
```

The output should be something similar to:

```text
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

Additionally, to check if your GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:

```text
import torch
torch.cuda.is_available()
```

