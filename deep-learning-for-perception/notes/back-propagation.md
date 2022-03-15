# Activation Function

## Activation Functions

Activation function or transfer function is used to determine the output of a neuron or node. It is a mathematical gate in between the input feeding the current neuron and its output going to the next layer.

![](<../../images/image (78).png>)

In deep learning, we commonly use non-linear Activation Function include

* Sigmoid: Output limit to \[0 1]. But gives gradient vanishing problem, not used anymore
* ReLU(rectified linear unit): most commonly used in CNN (hidden layers)
* Others: Tanh, Leaky ReLU, Maxout...

![Image from MIT Deeplearning Lecture](<../../images/image (83).png>)

![Cheat sheet of commonly used Activation Function](<../../images/image (77).png>)

## Output Activation Function

These functions are transformations we apply to vectors coming out from CNNs ( s ) before the loss computation. \[ [reference ](https://gombru.github.io/2018/05/23/cross\_entropy\_loss/)]

**Sigmoid**

It squashes a vector in the range (0, 1). It is applied independently to each element of ss sisi. It’s also called **logistic function**.

![](https://gombru.github.io/assets/cross\_entropy\_loss/sigmoid.png)[![](https://latex.codecogs.com/gif.latex?f%28s\_%7Bi%7D%29\&space;=\&space;%5Cfrac%7B1%7D%7B1\&space;+\&space;e%5E%7B-s\_%7Bi%7D%7D%7D)](https://www.codecogs.com/eqnedit.php?latex=f%28s\_{i}%29\&space;=\&space;\frac{1}{1\&space;+\&space;e^{-s\_{i\}}})

**Softmax**

Softmax it’s a function, not a loss. It squashes a vector in the range (0, 1) and all the resulting elements add up to 1. It is applied to the output scores ss. As elements represent a class, they can be interpreted as class probabilities.\
The Softmax function cannot be applied independently to each sisi, since it depends on all elements of ss. For a given class sisi, the Softmax function can be computed as:[![](https://latex.codecogs.com/gif.latex?f%28s%29\_%7Bi%7D\&space;=\&space;%5Cfrac%7Be%5E%7Bs\_%7Bi%7D%7D%7D%7B%5Csum\_%7Bj%7D%5E%7BC%7D\&space;e%5E%7Bs\_%7Bj%7D%7D%7D)](https://www.codecogs.com/eqnedit.php?latex=f%28s%29\_{i}\&space;=\&space;\frac{e^{s\_{i\}}}{\sum\_{j}^{C}\&space;e^{s\_{j\}}})

Where sjsj are the scores inferred by the net for each class in CC. Note that the Softmax activation for a class sisi depends on all the scores in ss.

###
