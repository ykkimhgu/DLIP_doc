# PyTorch

## What  is PyTorch

An open source machine learning framework that accelerates the path from research prototyping to production deployment. It’s a Python-based scientific computing package targeted at two sets of audiences:

* A replacement for NumPy to use the power of GPUs
* a deep learning research platform that provides maximum flexibility and speed

{% embed url="https://pytorch.org/features/" caption="" %}

## How to Install

Select your preferences and run the install command. Please ensure that you have **met the prerequisites below \(e.g., numpy\)**

\*\*\*\*[**https://pytorch.org/get-started/locally/**](https://pytorch.org/get-started/locally/)\*\*\*\*

1. Install CUDA or  check the installed GPU CUDA version

```text
nvcc --version
```

1. Install Anaconda. After installation, run **Anaconda Prompt**
2. Make a new virtual environment e.g torch16. In the \(base\) of Anaconda Prompt, create a new environment.

```text
conda env list
conda create --name torch16
activate torch16
```

1. Install  Python, Numpy, Panda and other prerequisite. Also, install necessary IDE \(Jupyter Notebook, Visual Studio Code etc.\)

> Check Python version. Need to have Python 3.x on Windows. `python --version`
>
> Check the list of packages installed in the environment`conda list`

1. Select your preferences and run the install command in your environment. The command can be found in  [**https://pytorch.org/get-started/locally/**](https://pytorch.org/get-started/locally/)  _\*\*_

**`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`**

1. **Install torchvision**. This package consists of popular datasets, model architectures, and common image transformations for computer vision. 

`conda install torchvision -c pytorch`

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

