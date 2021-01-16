# How to use

## How to run Jupyter Notebook

{% embed url="https://jupyter.readthedocs.io/en/latest/running.html" caption="" %}

## How to run Jupyter Lab

## How to run Jupyter Notebook in a virtual environment

## Where does it save ImageNet data?

## How to debug in Jupyter Notebook

[For reference link click here](https://medium.com/@chrieke/jupyter-tips-and-tricks-994fdddb2057#:~:text=The%20easiest%20way%20to%20debug,line%20that%20threw%20the%20error.)

* **%debug**

  The easiest way to debug a Jupyter notebook is to use the **%debug magic** command. Whenever you encounter an error or exception, just open a new notebook cell, type %debug and run the cell. This will open a command line where you can test your code and inspect all variables right up to the line that threw the error.

  Type **“n”** and hit Enter to run the **next** **line** of code \(The → arrow shows you the current position\). Use **“c”** to **continue** until the next breakpoint. **“q”** **quits** the debugger and code execution.

![Image for post](https://miro.medium.com/max/583/1*IiFphbhDWjmrUPwgUBgBhw.png)

* **iPython debugger**

Import it and use set\_trace\(\) anywhere in your notebook to create one or multiple breakpoints. When executing a cell, it will stop at the first breakpoint and open the command line for code inspection.

```text
from IPython.core.debugger import set_trace
set_trace()
```

![Image for post](https://miro.medium.com/max/654/1*WXn1k-GZvyiZoqCYGplqEQ.png)

* [**“PixieDebugger”**](https://medium.com/ibm-watson-data-lab/the-visual-python-debugger-for-jupyter-notebooks-youve-always-wanted-761713babc62)

  To invoke the PixieDebugger for a specific cell, simply add the **%%pixie\_debugger** magic at the top of the cell and run it.

&gt; _**Editor’s note:**_ PixieDebugger currently only works with classic Jupyter Notebooks; JupyterLab is not yet supported.

_**Note:**_ As a prerequisite, install PixieDust using the following pip command: `pip install pixiedust`. You’ll also need to import it into its own cell: `import pixiedust`.

```text
%%pixie_debugger
import random
def find_max (values):
    max = 0
    for val in values:
        if val > max:
            max = val
    return max
find_max(random.sample(range(100), 10))
```

