# Continuous Space RL

## Introduction

Discrete vs Continuous

* Discrete:
  * Grid-based world, finite number of states and actions.  
  * Chess grids, 
* Continuous Actions
  * real physical environment. robot control,  positions 

If there are a very large number of actions, TD control-SarsaMax needs to iterate for all those possible numbers of actions, which would increase the calculation load.

### Discretization: rounding to finite numbers

Discretize into finite states uniformly or non-uniformly. Example is an occupancy grid with equal size grids or non-uniform grid size

#### Tile Coding

#### Coarse Coding

* Use sparse data
* Narrow generalization
* Broad generalization
* Asymmetric generalization

### Function Approximation

Use parameters to shape the function that approximates the continuous value-state functions

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28154%29.png)

#### Linear Function Approximation

* Define the cost function of error \(approx function - true function\)
* And minimize the cost function using gradient descent.
* Only for linear relationship between the input and output

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28151%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28156%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28153%29.png)

Action-Vector approximation

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28160%29.png)

#### Kernel Functions / Feature Transformation

If the relationship is non-linear? Pass the relationship\( \) to a non-linear function, also known as activation function as in neural network.

* E.g. Radial Basis Functions

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28159%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28158%29.png)

