# Python with VS Code

## Python with VS Code

We will learn how to program Python \*.py in VS Code.

We want to make sure VSCode recognizes the Anaconda installation so it can access installed packages for autocomplete, syntax highlighting, and error checking.

### Prerequisite

* Anacond and Conda Virtual Environment : e.g. py37
  * [See Here for instruction](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning)
*   Python and Numpy Installed by Conda

    ​

## 1. Install Python Extension

* Open VS Code
* Press Extension Icon or Press < `Ctrl` +`Shift` +`X`>
* Install Python (Microsoft)

![image](https://user-images.githubusercontent.com/38373000/162184019-8d5b04af-a04a-486c-9e20-4786474e3c99.png)

## 2. Programming Python in VSCode

### Open Working Folder in VS Code

Create a new test folder

* Example: `\Tutorial_pythonOpenCV`

Open the folder in VS Code

![image](https://user-images.githubusercontent.com/38373000/162183686-3b7a6a12-adff-4fef-aa59-a9f0b3a9372c.png)

### Select Interpreter

Let's select the virtual environment for this tutorial.

* Assumed we already have created Conda Environment.
* [See Here for Conda Environment Creation](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning)

Press `F1` > 'Python: Select Interpreter' > Choose `py37` as the Interpreter

![image](https://user-images.githubusercontent.com/38373000/162185395-9265cb2e-2441-41d5-9af8-3ad05df0938f.png)

### Run Test Code

(1) Create a new file

(2) Name the file as `TU_pytest.py`

(3) Program the code

```python
import numpy as np
a = np.array([1,2,3])
print(a*a)
```

(4) Run Python File

(5) It should give the output as

`[1 4 9]`

![image](https://user-images.githubusercontent.com/38373000/162189129-3617e587-7263-45e3-8059-290472d36fd0.png)

## Further Reading

[How to program Jupyter and CoLab(Notebook) in VS Code](https://ykkim.gitbook.io/dlip/installation-guide/ide/vscode/notebook-with-vscode)
