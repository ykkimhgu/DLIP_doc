# CoLab and Jupyter with VS Code



https://opensourceoptions.com/blog/jupyter-notebooks-in-visual-studio-code/



It is simple to get Jupyter or Colab notebooks running inside Visual Studio Code.

1. Install the necessary Jupyter notebook extensions **in VS Code**
2. Install the Jupyter module **in Conda Python Environment**
3. Open or Make a .ipynb file
4. Run the code





## 1. Install VSC Extensions to Enable Jupyter Notebooks

Once you have VSC installed you’ll just need to install a few extensions for VSC to support Jupyter notebooks. You’ll need to install six extensions.

- Python (author: Microsoft)
- Pylance (author: Microsoft)
- Live Server (author: Ritwick Dey)
- Jupyter (author: Microsoft)
- Jupyter Keymap (author: Microsoft)
- Jupter Notebook Renderers (author: Microsoft)




## 2.  Install the Jupyter Module in  Conda Python Environment

Need an environment to install the `jupyter` module 

* If Conda Environment exists, you can skip this 

```C
conda create -n py37 python=3.7
conda activate py37
```



* Install the `jupyter` module.

```C
conda install jupyter
```



## 3. Open or Make a .ipynb file and Run

Lets download *.ipynb from Colab. 

* [Click here for a Test Colab Code](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_colab_vscode.ipynb)



Download the file as *.ipynb.

![image](https://user-images.githubusercontent.com/38373000/162196283-0569091f-e0b3-4d53-83a8-63cfcf6dea1d.png)



Open the downloaded file in VSCODE  and Run

* Select Python Interpreter as `py37` or where you have installed Jupyter

![image](https://user-images.githubusercontent.com/38373000/162197036-96afbe6f-2610-4b70-8b5d-011668870e70.png)

