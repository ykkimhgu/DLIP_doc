# Tutorial: Python Numpy

## Preparation

### Installation of Python and Numpy in Visual Studio Code

Read here to install Python and Numpy in Visual Studio Code

{% embed url="https://ykkim.gitbook.io/dlip/image-processing/tutorial/tutorial-installation-for-py-opencv" %}

###

## Tutorials

Skip this if you already know about Python programming

Recommended tutorial lists for Python and Numpy

#### Python Tutorial  <a href="#python-tutorial" id="python-tutorial"></a>

In-Class Python Tutorial: [Learning Python in 60 minutes\_DLIP](https://colab.research.google.com/drive/1W0dVnmMUF_Yj-mb7B0tq5kl4x5yWHtff)

#### Numpy Tutorial <a href="#numpy-tutorial" id="numpy-tutorial"></a>

NumPy Tutorial: [Learning Numpy in 60 minutes\_DLIP](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_PythonNumpy/Tutorial_Numpy_2022.ipynb)

#### **For More Tutorials**

{% embed url="https://ykkim.gitbook.io/dlip/programming/python#tutorial" %}



## Exercise&#x20;

### Create an MLP to solve an XOR problem (Numpy)



Open or Download the exercise  colab notebook file:   [Tutorial\_Numpy\_MLP\_XOR.ipynb](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial_PythonNumpy)



<details>

<summary>Solution</summary>

```py

# MPL structure & initialization
W0 = 2*np.random.randn(3,4)-1
W1 = 2*np.random.randn(4,1)-1
 
# training
eta = 1; # learning rate
iterNo = 1000
for i in range(0, iterNo):
    # forward direction
    S0 = np.matmul(X, W0)                                   # S0 = X*W0;
    L1 = np.divide(1,(1+np.exp(-S0)))                       # L1 = 1./(1+exp(-S0));
    S1 = np.matmul(L1, W1)                                  # S1 = L1*W1;
    Yh = np.divide(1,(1+np.exp(-S1)))                       # Yh = 1./(1+exp(-S1));
    #print(Yh.shape)
    #print(Y.shape)
    # error backpropagation
    dE_dS1 = np.multiply(np.multiply(-(Y-Yh), Yh), 1-Yh)    # dE_dS1 = -(Y-Yh).*Yh.*(1-Yh)
    
    #print(W1.shape)
    #print(dE_dS1.shape)

    dE_dL1 = np.matmul(dE_dS1, np.transpose(W1))            # dE_dL1 = dE_dS1 *W1'
    dE_dS0 = np.multiply(np.multiply(dE_dL1, L1), 1-L1)     # dE_dS0 = dE_dL1.*L1.*(1-L1)
    # gradient descent algorithm
    dE_dS0 = np.multiply(np.multiply(dE_dL1, L1), 1-L1)     # dE_dS0 = dE_dL1.*L1.*(1-L1)
    dE_dW1 = np.matmul(np.transpose(L1), dE_dS1)            # dE_dW1 = L1'*dE_dS1
    dE_dW0 = np.matmul(np.transpose(X),  dE_dS0)            # dE_dW0 = X'*dE_dS0
    W1 = W1 - eta*dE_dW1                                    # W1 = W1-eta*dE_dW1
    W0 = W0 - eta*dE_dW0                                    # W0 = W0 - eta*dE_dW0
    if (not(i%100)):
        print("iter = {0}, Yh = {1}\n".format(i, Yh.tolist()))
```



</details>

